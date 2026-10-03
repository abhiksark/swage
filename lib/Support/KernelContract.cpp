//===- KernelContract.cpp - Physical kernel ABI contract -----------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Support/KernelContract.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

using namespace mlir;

namespace mlir::swage {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(message);
}

llvm::StringRef stringify(KernelBackend backend) {
  switch (backend) {
  case KernelBackend::CUDA:
    return "cuda";
  case KernelBackend::CPU:
    return "cpu";
  }
  llvm_unreachable("unknown kernel backend");
}

llvm::StringRef stringify(KernelLaunchModel model) {
  switch (model) {
  case KernelLaunchModel::SPMDGrid:
    return "spmd-grid";
  case KernelLaunchModel::HostCall:
    return "host-call";
  }
  llvm_unreachable("unknown kernel launch model");
}

llvm::StringRef stringify(KernelArgumentKind kind) {
  switch (kind) {
  case KernelArgumentKind::Pointer:
    return "ptr";
  case KernelArgumentKind::I1:
    return "i1";
  case KernelArgumentKind::I8:
    return "i8";
  case KernelArgumentKind::I16:
    return "i16";
  case KernelArgumentKind::I32:
    return "i32";
  case KernelArgumentKind::I64:
    return "i64";
  case KernelArgumentKind::F16:
    return "f16";
  case KernelArgumentKind::BF16:
    return "bf16";
  case KernelArgumentKind::F32:
    return "f32";
  case KernelArgumentKind::F64:
    return "f64";
  }
  llvm_unreachable("unknown kernel argument kind");
}

llvm::StringRef stringify(KernelArgumentOrigin origin) {
  switch (origin) {
  case KernelArgumentOrigin::User:
    return "user";
  case KernelArgumentOrigin::Derived:
    return "derived";
  case KernelArgumentOrigin::Plan:
    return "plan";
  case KernelArgumentOrigin::Scratch:
    return "scratch";
  }
  llvm_unreachable("unknown kernel argument origin");
}

llvm::StringRef stringify(KernelArgumentAccess access) {
  switch (access) {
  case KernelArgumentAccess::Read:
    return "read";
  case KernelArgumentAccess::Write:
    return "write";
  case KernelArgumentAccess::ReadWrite:
    return "readwrite";
  }
  llvm_unreachable("unknown kernel argument access");
}

llvm::Expected<KernelBackend> parseBackend(llvm::StringRef value) {
  if (value == "cuda")
    return KernelBackend::CUDA;
  if (value == "cpu")
    return KernelBackend::CPU;
  return invalid("unknown kernel backend '" + value + "'");
}

llvm::Expected<KernelLaunchModel> parseLaunchModel(llvm::StringRef value) {
  if (value == "spmd-grid")
    return KernelLaunchModel::SPMDGrid;
  if (value == "host-call")
    return KernelLaunchModel::HostCall;
  return invalid("unknown kernel launch model '" + value + "'");
}

llvm::Expected<KernelArgumentKind> parseKind(llvm::StringRef value) {
  if (value == "ptr")
    return KernelArgumentKind::Pointer;
  if (value == "i1")
    return KernelArgumentKind::I1;
  if (value == "i8")
    return KernelArgumentKind::I8;
  if (value == "i16")
    return KernelArgumentKind::I16;
  if (value == "i32")
    return KernelArgumentKind::I32;
  if (value == "i64")
    return KernelArgumentKind::I64;
  if (value == "f16")
    return KernelArgumentKind::F16;
  if (value == "bf16")
    return KernelArgumentKind::BF16;
  if (value == "f32")
    return KernelArgumentKind::F32;
  if (value == "f64")
    return KernelArgumentKind::F64;
  return invalid("unknown kernel argument kind '" + value + "'");
}

llvm::Expected<KernelArgumentOrigin> parseOrigin(llvm::StringRef value) {
  if (value == "user")
    return KernelArgumentOrigin::User;
  if (value == "derived")
    return KernelArgumentOrigin::Derived;
  if (value == "plan")
    return KernelArgumentOrigin::Plan;
  if (value == "scratch")
    return KernelArgumentOrigin::Scratch;
  return invalid("unknown kernel argument origin '" + value + "'");
}

llvm::Expected<KernelArgumentAccess> parseAccess(llvm::StringRef value) {
  if (value == "read")
    return KernelArgumentAccess::Read;
  if (value == "write")
    return KernelArgumentAccess::Write;
  if (value == "readwrite")
    return KernelArgumentAccess::ReadWrite;
  return invalid("unknown kernel argument access '" + value + "'");
}

llvm::Error rejectUnknownFields(const llvm::json::Object &fields,
                                llvm::ArrayRef<llvm::StringRef> allowed,
                                const llvm::Twine &where) {
  for (const auto &field : fields) {
    llvm::StringRef name = field.first;
    if (!llvm::is_contained(allowed, name))
      return invalid(where + " has unknown field '" + name + "'");
  }
  return llvm::Error::success();
}

llvm::Error rejectUnknownFields(DictionaryAttr fields,
                                llvm::ArrayRef<llvm::StringRef> allowed,
                                const llvm::Twine &where) {
  for (NamedAttribute field : fields) {
    llvm::StringRef name = field.getName().getValue();
    if (!llvm::is_contained(allowed, name))
      return invalid(where + " has unknown field '" + name + "'");
  }
  return llvm::Error::success();
}
llvm::Expected<KernelLaunch> parseJSONLaunch(const llvm::json::Object &object) {
  if (llvm::Error error =
          rejectUnknownFields(object, {"model", "block"}, "kernel launch"))
    return std::move(error);
  std::optional<llvm::StringRef> modelText = object.getString("model");
  if (!modelText)
    return invalid("kernel launch requires a string model");
  llvm::Expected<KernelLaunchModel> model = parseLaunchModel(*modelText);
  if (!model)
    return model.takeError();

  KernelLaunch launch;
  launch.model = *model;
  launch.block = std::nullopt;
  if (*model == KernelLaunchModel::HostCall) {
    if (object.get("block"))
      return invalid("host-call launch must not have block");
    return launch;
  }

  const llvm::json::Array *block = object.getArray("block");
  if (!block || block->size() != 3)
    return invalid("spmd-grid launch requires three block dimensions");
  std::array<int32_t, 3> dimensions;
  for (size_t index = 0; index < block->size(); ++index) {
    std::optional<int64_t> dimension = (*block)[index].getAsInteger();
    if (!dimension || *dimension < std::numeric_limits<int32_t>::min() ||
        *dimension > std::numeric_limits<int32_t>::max())
      return invalid("kernel launch block dimensions must be i32 integers");
    dimensions[index] = static_cast<int32_t>(*dimension);
  }
  launch.block = dimensions;
  return launch;
}

llvm::Expected<KernelLaunch> parseAttrLaunch(DictionaryAttr object) {
  if (llvm::Error error =
          rejectUnknownFields(object, {"model", "block"}, "kernel launch"))
    return std::move(error);
  auto modelText = object.getAs<StringAttr>("model");
  if (!modelText)
    return invalid("kernel launch requires a string model");
  llvm::Expected<KernelLaunchModel> model =
      parseLaunchModel(modelText.getValue());
  if (!model)
    return model.takeError();

  KernelLaunch launch;
  launch.model = *model;
  launch.block = std::nullopt;
  if (*model == KernelLaunchModel::HostCall) {
    if (object.get("block"))
      return invalid("host-call launch must not have block");
    return launch;
  }

  auto block = object.getAs<DenseI32ArrayAttr>("block");
  if (!block || block.size() != 3)
    return invalid("spmd-grid launch requires three i32 block dimensions");
  std::array<int32_t, 3> dimensions;
  llvm::copy(block.asArrayRef(), dimensions.begin());
  launch.block = dimensions;
  return launch;
}

llvm::Expected<KernelArgument> parseJSONArgument(const llvm::json::Value &value,
                                                 size_t index) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid("argument " + llvm::Twine(index) + " must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *object, {"kind", "origin", "source_index", "key", "access"},
          "argument " + llvm::Twine(index)))
    return std::move(error);

  std::optional<llvm::StringRef> kindText = object->getString("kind");
  std::optional<llvm::StringRef> originText = object->getString("origin");
  if (!kindText || !originText)
    return invalid("argument " + llvm::Twine(index) +
                   " requires string fields 'kind' and 'origin'");
  llvm::Expected<KernelArgumentKind> kind = parseKind(*kindText);
  if (!kind)
    return kind.takeError();
  llvm::Expected<KernelArgumentOrigin> origin = parseOrigin(*originText);
  if (!origin)
    return origin.takeError();

  KernelArgument argument{*kind, *origin, std::nullopt, "", std::nullopt};
  if (*origin == KernelArgumentOrigin::User) {
    std::optional<int64_t> sourceIndex = object->getInteger("source_index");
    if (!sourceIndex || *sourceIndex < 0 ||
        static_cast<uint64_t>(*sourceIndex) >
            std::numeric_limits<uint32_t>::max())
      return invalid("user argument " + llvm::Twine(index) +
                     " requires a nonnegative u32 source_index");
    if (object->get("key"))
      return invalid("user argument " + llvm::Twine(index) +
                     " must not have a key");
    argument.sourceIndex = static_cast<uint32_t>(*sourceIndex);
  } else {
    std::optional<llvm::StringRef> key = object->getString("key");
    if (!key || key->empty())
      return invalid("non-user argument " + llvm::Twine(index) +
                     " requires a nonempty string key");
    if (object->get("source_index"))
      return invalid("non-user argument " + llvm::Twine(index) +
                     " must not have a source_index");
    argument.key = key->str();
  }

  if (*kind == KernelArgumentKind::Pointer) {
    std::optional<llvm::StringRef> accessText = object->getString("access");
    if (!accessText)
      return invalid("pointer argument " + llvm::Twine(index) +
                     " requires a string access");
    llvm::Expected<KernelArgumentAccess> access = parseAccess(*accessText);
    if (!access)
      return access.takeError();
    argument.access = *access;
  } else if (object->get("access")) {
    return invalid("scalar argument " + llvm::Twine(index) +
                   " must not have access");
  }
  return argument;
}

llvm::Expected<KernelArgument> parseAttrArgument(DictionaryAttr object,
                                                 size_t index) {
  if (llvm::Error error = rejectUnknownFields(
          object, {"kind", "origin", "source_index", "key", "access"},
          "argument " + llvm::Twine(index)))
    return std::move(error);
  auto kindText = object.getAs<StringAttr>("kind");
  auto originText = object.getAs<StringAttr>("origin");
  if (!kindText || !originText)
    return invalid("argument " + llvm::Twine(index) +
                   " requires string attributes 'kind' and 'origin'");
  llvm::Expected<KernelArgumentKind> kind = parseKind(kindText.getValue());
  if (!kind)
    return kind.takeError();
  llvm::Expected<KernelArgumentOrigin> origin =
      parseOrigin(originText.getValue());
  if (!origin)
    return origin.takeError();

  KernelArgument argument{*kind, *origin, std::nullopt, "", std::nullopt};
  if (*origin == KernelArgumentOrigin::User) {
    auto sourceIndex = object.getAs<IntegerAttr>("source_index");
    if (!sourceIndex || !sourceIndex.getType().isSignlessInteger(64) ||
        sourceIndex.getInt() < 0 ||
        static_cast<uint64_t>(sourceIndex.getInt()) >
            std::numeric_limits<uint32_t>::max())
      return invalid("user argument " + llvm::Twine(index) +
                     " requires a nonnegative i64 source_index");
    if (object.get("key"))
      return invalid("user argument " + llvm::Twine(index) +
                     " must not have a key");
    argument.sourceIndex = static_cast<uint32_t>(sourceIndex.getInt());
  } else {
    auto key = object.getAs<StringAttr>("key");
    if (!key || key.getValue().empty())
      return invalid("non-user argument " + llvm::Twine(index) +
                     " requires a nonempty string key");
    if (object.get("source_index"))
      return invalid("non-user argument " + llvm::Twine(index) +
                     " must not have a source_index");
    argument.key = key.getValue().str();
  }

  if (*kind == KernelArgumentKind::Pointer) {
    auto accessText = object.getAs<StringAttr>("access");
    if (!accessText)
      return invalid("pointer argument " + llvm::Twine(index) +
                     " requires a string access");
    llvm::Expected<KernelArgumentAccess> access =
        parseAccess(accessText.getValue());
    if (!access)
      return access.takeError();
    argument.access = *access;
  } else if (object.get("access")) {
    return invalid("scalar argument " + llvm::Twine(index) +
                   " must not have access");
  }
  return argument;
}

} // namespace

KernelArgument KernelArgument::userPointer(uint32_t sourceIndex,
                                           KernelArgumentAccess access) {
  return {KernelArgumentKind::Pointer, KernelArgumentOrigin::User, sourceIndex,
          "", access};
}

KernelArgument KernelArgument::userScalar(KernelArgumentKind kind,
                                          uint32_t sourceIndex) {
  return {kind, KernelArgumentOrigin::User, sourceIndex, "", std::nullopt};
}

KernelArgument KernelArgument::keyedPointer(KernelArgumentOrigin origin,
                                            llvm::StringRef key,
                                            KernelArgumentAccess access) {
  return {KernelArgumentKind::Pointer, origin, std::nullopt, key.str(), access};
}

KernelArgument KernelArgument::keyedScalar(KernelArgumentKind kind,
                                           KernelArgumentOrigin origin,
                                           llvm::StringRef key) {
  return {kind, origin, std::nullopt, key.str(), std::nullopt};
}

bool KernelArgument::operator==(const KernelArgument &other) const {
  return kind == other.kind && origin == other.origin &&
         sourceIndex == other.sourceIndex && key == other.key &&
         access == other.access;
}
bool KernelLaunch::operator==(const KernelLaunch &other) const {
  return model == other.model && block == other.block;
}

bool KernelContract::operator==(const KernelContract &other) const {
  return version == other.version && backend == other.backend &&
         entry == other.entry && launch == other.launch &&
         arguments == other.arguments;
}

llvm::Error validateKernelContract(const KernelContract &contract) {
  if (contract.version != kernelContractVersion)
    return invalid("unsupported kernel contract version " +
                   llvm::Twine(contract.version));
  if (contract.entry.empty())
    return invalid("kernel contract entry must not be empty");

  switch (contract.backend) {
  case KernelBackend::CUDA:
    if (contract.launch.model != KernelLaunchModel::SPMDGrid)
      return invalid("CUDA kernel contract requires spmd-grid launch");
    break;
  case KernelBackend::CPU:
    if (contract.launch.model != KernelLaunchModel::HostCall)
      return invalid("CPU kernel contract requires host-call launch");
    break;
  default:
    return invalid("kernel contract has an invalid backend");
  }

  switch (contract.launch.model) {
  case KernelLaunchModel::SPMDGrid:
    if (!contract.launch.block)
      return invalid("spmd-grid launch requires block");
    for (int32_t dimension : *contract.launch.block) {
      if (dimension <= 0)
        return invalid("kernel contract block dimensions must be positive");
    }
    break;
  case KernelLaunchModel::HostCall:
    if (contract.launch.block)
      return invalid("host-call launch must not have block");
    break;
  default:
    return invalid("kernel contract has an invalid launch model");
  }

  llvm::DenseSet<uint32_t> sourceIndexes;
  llvm::DenseSet<llvm::StringRef> keys;
  for (const auto &[index, argument] : llvm::enumerate(contract.arguments)) {
    switch (argument.kind) {
    case KernelArgumentKind::Pointer:
    case KernelArgumentKind::I1:
    case KernelArgumentKind::I8:
    case KernelArgumentKind::I16:
    case KernelArgumentKind::I32:
    case KernelArgumentKind::I64:
    case KernelArgumentKind::F16:
    case KernelArgumentKind::BF16:
    case KernelArgumentKind::F32:
    case KernelArgumentKind::F64:
      break;
    default:
      return invalid("argument " + llvm::Twine(index) + " has an invalid kind");
    }
    switch (argument.origin) {
    case KernelArgumentOrigin::User:
    case KernelArgumentOrigin::Derived:
    case KernelArgumentOrigin::Plan:
    case KernelArgumentOrigin::Scratch:
      break;
    default:
      return invalid("argument " + llvm::Twine(index) +
                     " has an invalid origin");
    }
    if (argument.access) {
      switch (*argument.access) {
      case KernelArgumentAccess::Read:
      case KernelArgumentAccess::Write:
      case KernelArgumentAccess::ReadWrite:
        break;
      default:
        return invalid("argument " + llvm::Twine(index) +
                       " has invalid access");
      }
    }
    if (argument.origin == KernelArgumentOrigin::User) {
      if (!argument.sourceIndex || !argument.key.empty())
        return invalid("user argument " + llvm::Twine(index) +
                       " must bind only by source_index");
      if (!sourceIndexes.insert(*argument.sourceIndex).second)
        return invalid("duplicate user source_index " +
                       llvm::Twine(*argument.sourceIndex));
    } else {
      if (argument.sourceIndex || argument.key.empty())
        return invalid("non-user argument " + llvm::Twine(index) +
                       " must bind only by key");
      if (!keys.insert(argument.key).second)
        return invalid("duplicate non-user binding key '" + argument.key + "'");
    }
    if (argument.kind == KernelArgumentKind::Pointer && !argument.access)
      return invalid("pointer argument " + llvm::Twine(index) +
                     " requires access");
    if (argument.kind != KernelArgumentKind::Pointer && argument.access)
      return invalid("scalar argument " + llvm::Twine(index) +
                     " must not have access");
  }
  return llvm::Error::success();
}

llvm::Expected<KernelContract> parseKernelContractJSON(llvm::StringRef text) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("kernel contract must be a JSON object");
  if (llvm::Error error = rejectUnknownFields(
          *object, {"version", "backend", "entry", "launch", "arguments"},
          "kernel contract"))
    return std::move(error);

  std::optional<int64_t> version = object->getInteger("version");
  std::optional<llvm::StringRef> backendText = object->getString("backend");
  std::optional<llvm::StringRef> entry = object->getString("entry");
  const llvm::json::Object *launchObject = object->getObject("launch");
  const llvm::json::Array *arguments = object->getArray("arguments");
  if (!version || !backendText || !entry || !launchObject || !arguments)
    return invalid(
        "kernel contract requires version, backend, entry, launch, and "
        "arguments");

  llvm::Expected<KernelBackend> backend = parseBackend(*backendText);
  if (!backend)
    return backend.takeError();
  llvm::Expected<KernelLaunch> launch = parseJSONLaunch(*launchObject);
  if (!launch)
    return launch.takeError();

  KernelContract contract;
  contract.version = *version;
  contract.backend = *backend;
  contract.entry = entry->str();
  contract.launch = *launch;
  contract.arguments.reserve(arguments->size());
  for (const auto &[index, argumentValue] : llvm::enumerate(*arguments)) {
    llvm::Expected<KernelArgument> argument =
        parseJSONArgument(argumentValue, index);
    if (!argument)
      return argument.takeError();
    contract.arguments.push_back(std::move(*argument));
  }
  if (llvm::Error error = validateKernelContract(contract))
    return std::move(error);
  return contract;
}

std::string serializeKernelContractJSON(const KernelContract &contract) {
  std::string storage;
  llvm::raw_string_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("version", contract.version);
    json.attribute("backend", stringify(contract.backend));
    json.attribute("entry", contract.entry);
    json.attributeObject("launch", [&] {
      json.attribute("model", stringify(contract.launch.model));
      if (contract.launch.block) {
        json.attributeArray("block", [&] {
          for (int32_t dimension : *contract.launch.block)
            json.value(dimension);
        });
      }
    });
    json.attributeArray("arguments", [&] {
      for (const KernelArgument &argument : contract.arguments) {
        json.object([&] {
          json.attribute("kind", stringify(argument.kind));
          json.attribute("origin", stringify(argument.origin));
          if (argument.origin == KernelArgumentOrigin::User)
            json.attribute("source_index", *argument.sourceIndex);
          else
            json.attribute("key", argument.key);
          if (argument.kind == KernelArgumentKind::Pointer)
            json.attribute("access", stringify(*argument.access));
        });
      }
    });
  });
  json.flush();
  return storage;
}

std::string digestKernelContract(const KernelContract &contract) {
  std::string canonical = serializeKernelContractJSON(contract);
  std::array<uint8_t, 32> digest = llvm::SHA256::hash(
      llvm::arrayRefFromStringRef(llvm::StringRef(canonical)));
  return llvm::toHex(digest, /*LowerCase=*/true);
}

DictionaryAttr buildKernelContractAttr(Builder &builder,
                                       const KernelContract &contract) {
  SmallVector<NamedAttribute> launchFields;
  launchFields.emplace_back(
      builder.getStringAttr("model"),
      builder.getStringAttr(stringify(contract.launch.model)));
  if (contract.launch.block)
    launchFields.emplace_back(
        builder.getStringAttr("block"),
        builder.getDenseI32ArrayAttr(*contract.launch.block));

  SmallVector<Attribute> arguments;
  arguments.reserve(contract.arguments.size());
  for (const KernelArgument &argument : contract.arguments) {
    SmallVector<NamedAttribute> fields;
    fields.emplace_back(builder.getStringAttr("kind"),
                        builder.getStringAttr(stringify(argument.kind)));
    fields.emplace_back(builder.getStringAttr("origin"),
                        builder.getStringAttr(stringify(argument.origin)));
    if (argument.origin == KernelArgumentOrigin::User)
      fields.emplace_back(builder.getStringAttr("source_index"),
                          builder.getI64IntegerAttr(*argument.sourceIndex));
    else
      fields.emplace_back(builder.getStringAttr("key"),
                          builder.getStringAttr(argument.key));
    if (argument.kind == KernelArgumentKind::Pointer)
      fields.emplace_back(builder.getStringAttr("access"),
                          builder.getStringAttr(stringify(*argument.access)));
    arguments.push_back(builder.getDictionaryAttr(fields));
  }
  return builder.getDictionaryAttr(
      {{builder.getStringAttr("version"),
        builder.getI64IntegerAttr(contract.version)},
       {builder.getStringAttr("backend"),
        builder.getStringAttr(stringify(contract.backend))},
       {builder.getStringAttr("entry"), builder.getStringAttr(contract.entry)},
       {builder.getStringAttr("launch"),
        builder.getDictionaryAttr(launchFields)},
       {builder.getStringAttr("arguments"), builder.getArrayAttr(arguments)}});
}

llvm::Expected<KernelContract> parseKernelContractAttr(Attribute attribute) {
  auto object = dyn_cast_or_null<DictionaryAttr>(attribute);
  if (!object)
    return invalid("swage.kernel_contract must be a dictionary attribute");
  if (llvm::Error error = rejectUnknownFields(
          object, {"version", "backend", "entry", "launch", "arguments"},
          "kernel contract"))
    return std::move(error);

  auto version = object.getAs<IntegerAttr>("version");
  auto backendText = object.getAs<StringAttr>("backend");
  auto entry = object.getAs<StringAttr>("entry");
  auto launchObject = object.getAs<DictionaryAttr>("launch");
  auto arguments = object.getAs<ArrayAttr>("arguments");
  if (!version || !version.getType().isSignlessInteger(64) || !backendText ||
      !entry || !launchObject || !arguments)
    return invalid("kernel contract requires i64 version, string backend, "
                   "string entry, dictionary launch, and array arguments");

  llvm::Expected<KernelBackend> backend = parseBackend(backendText.getValue());
  if (!backend)
    return backend.takeError();
  llvm::Expected<KernelLaunch> launch = parseAttrLaunch(launchObject);
  if (!launch)
    return launch.takeError();

  KernelContract contract;
  contract.version = version.getInt();
  contract.backend = *backend;
  contract.entry = entry.getValue().str();
  contract.launch = *launch;
  contract.arguments.reserve(arguments.size());
  for (const auto &[index, argumentAttr] : llvm::enumerate(arguments)) {
    auto argumentObject = dyn_cast<DictionaryAttr>(argumentAttr);
    if (!argumentObject)
      return invalid("argument " + llvm::Twine(index) +
                     " must be a dictionary attribute");
    llvm::Expected<KernelArgument> argument =
        parseAttrArgument(argumentObject, index);
    if (!argument)
      return argument.takeError();
    contract.arguments.push_back(std::move(*argument));
  }
  if (llvm::Error error = validateKernelContract(contract))
    return std::move(error);
  return contract;
}

SmallVector<Type>
getKernelArgumentTypes(MLIRContext *context,
                       llvm::ArrayRef<KernelArgument> arguments) {
  SmallVector<Type> types;
  types.reserve(arguments.size());
  for (const KernelArgument &argument : arguments) {
    switch (argument.kind) {
    case KernelArgumentKind::Pointer:
      types.push_back(LLVM::LLVMPointerType::get(context));
      break;
    case KernelArgumentKind::I1:
      types.push_back(IntegerType::get(context, 1));
      break;
    case KernelArgumentKind::I8:
      types.push_back(IntegerType::get(context, 8));
      break;
    case KernelArgumentKind::I16:
      types.push_back(IntegerType::get(context, 16));
      break;
    case KernelArgumentKind::I32:
      types.push_back(IntegerType::get(context, 32));
      break;
    case KernelArgumentKind::I64:
      types.push_back(IntegerType::get(context, 64));
      break;
    case KernelArgumentKind::F16:
      types.push_back(Float16Type::get(context));
      break;
    case KernelArgumentKind::BF16:
      types.push_back(BFloat16Type::get(context));
      break;
    case KernelArgumentKind::F32:
      types.push_back(Float32Type::get(context));
      break;
    case KernelArgumentKind::F64:
      types.push_back(Float64Type::get(context));
      break;
    }
  }
  return types;
}

llvm::Error validateKernelContractAgainstFunction(
    const KernelContract &contract, llvm::StringRef entry, FunctionType type) {
  if (llvm::Error error = validateKernelContract(contract))
    return error;
  if (contract.entry != entry)
    return invalid("kernel contract entry '" + contract.entry +
                   "' does not match function '" + entry + "'");
  if (type.getNumResults() != 0)
    return invalid("contract-bearing function must not return values");
  if (type.getNumInputs() != contract.arguments.size())
    return invalid("kernel contract argument count does not match function");

  for (auto [index, input] : llvm::enumerate(type.getInputs())) {
    bool matches = false;
    switch (contract.arguments[index].kind) {
    case KernelArgumentKind::Pointer:
      matches = isa<LLVM::LLVMPointerType>(input);
      break;
    case KernelArgumentKind::I1:
      matches = input.isSignlessInteger(1);
      break;
    case KernelArgumentKind::I8:
      matches = input.isSignlessInteger(8);
      break;
    case KernelArgumentKind::I16:
      matches = input.isSignlessInteger(16);
      break;
    case KernelArgumentKind::I32:
      matches = input.isSignlessInteger(32);
      break;
    case KernelArgumentKind::I64:
      matches = input.isSignlessInteger(64);
      break;
    case KernelArgumentKind::F16:
      matches = input.isF16();
      break;
    case KernelArgumentKind::BF16:
      matches = input.isBF16();
      break;
    case KernelArgumentKind::F32:
      matches = input.isF32();
      break;
    case KernelArgumentKind::F64:
      matches = input.isF64();
      break;
    }
    if (!matches)
      return invalid("kernel contract argument " + llvm::Twine(index) +
                     " kind does not match function type");
  }
  return llvm::Error::success();
}

llvm::Error validateCUDAKernelContract(const KernelContract &contract,
                                       llvm::StringRef entry, FunctionType type,
                                       llvm::ArrayRef<int32_t> requiredBlock) {
  if (llvm::Error error =
          validateKernelContractAgainstFunction(contract, entry, type))
    return error;
  if (contract.backend != KernelBackend::CUDA ||
      contract.launch.model != KernelLaunchModel::SPMDGrid ||
      !contract.launch.block)
    return invalid("compiled CUDA entry requires a CUDA spmd-grid contract");
  if (requiredBlock.size() != contract.launch.block->size() ||
      !llvm::equal(requiredBlock, *contract.launch.block))
    return invalid("kernel contract block does not match nvvm.reqntid");

  uint64_t threads = 1;
  for (int32_t dimension : *contract.launch.block) {
    if (threads > 1024 / static_cast<uint64_t>(dimension))
      return invalid("CUDA kernel contract block may contain at most 1024 "
                     "threads");
    threads *= static_cast<uint64_t>(dimension);
  }
  return llvm::Error::success();
}

llvm::Error validateHostKernelContract(const KernelContract &contract,
                                       llvm::StringRef entry,
                                       FunctionType type) {
  if (llvm::Error error =
          validateKernelContractAgainstFunction(contract, entry, type))
    return error;
  if (contract.backend != KernelBackend::CPU ||
      contract.launch.model != KernelLaunchModel::HostCall ||
      contract.launch.block)
    return invalid("compiled host entry requires a CPU host-call contract");
  return llvm::Error::success();
}

} // namespace mlir::swage
