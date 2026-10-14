// test/Dialect/Swage/invalid-roles.mlir
// RUN: swage-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{swage.role<values> requires a memref of rank one or two, got 'i32'}}
func.func @values_on_a_count(
    %values: i32 {swage.role = #swage.role<values>}) {
  return
}

// -----

// expected-error @below {{swage.role<output> requires a memref of rank one or two, got 'memref<?x?x?xf32>'}}
func.func @output_of_rank_three(
    %output: memref<?x?x?xf32> {swage.role = #swage.role<output>}) {
  return
}

// -----

// expected-error @below {{swage.role<offsets> requires a rank-one memref of signless integers, got 'memref<?xf32>'}}
func.func @float_offsets(
    %offsets: memref<?xf32> {swage.role = #swage.role<offsets>}) {
  return
}

// -----

// expected-error @below {{swage.role<value_count> requires a signless integer, got 'memref<?xi32>'}}
func.func @count_on_a_buffer(
    %value_count: memref<?xi32> {swage.role = #swage.role<value_count>}) {
  return
}

// -----

// expected-error @below {{swage.role<segment_count> requires a signless integer, got 'index'}}
func.func @index_count(
    %segment_count: index {swage.role = #swage.role<segment_count>}) {
  return
}

// -----

// expected-error @below {{swage.role<values> is declared by argument #0 and again by argument #2}}
func.func @two_values(
    %values: memref<?xf32> {swage.role = #swage.role<values>},
    %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
    %output: memref<?xf32> {swage.role = #swage.role<values>}) {
  return
}

// -----

// expected-error @below {{swage.role must be a #swage.role attribute, got unit}}
func.func @role_without_a_value(%values: memref<?xf32> {swage.role}) {
  return
}

// -----

// expected-error @below {{argument #0 carries 'swage.rank', which is not an argument attribute of the swage dialect; the dialect defines swage.role}}
func.func @unknown_argument_attribute(
    %values: memref<?xf32> {swage.rank = 1 : i64}) {
  return
}

// -----

func.func @unknown_role(
    // expected-error @below {{expected ::mlir::swage::ArgumentRole to be one of: values, offsets, output, value_count, segment_count, feature_count}}
    // expected-error @below {{failed to parse Swage_ArgumentRoleAttr parameter 'value'}}
    %values: memref<?xf32> {swage.role = #swage.role<input>}) {
  return
}

// -----

// expected-error @below {{swage.role<feature_count> requires a signless integer, got 'index'}}
func.func @index_feature_count(
    %feature_count: index {swage.role = #swage.role<feature_count>}) {
  return
}

// -----

// expected-error @below {{swage.role<feature_count> is declared by argument #0 and again by argument #1}}
func.func @two_feature_counts(
    %columns: i32 {swage.role = #swage.role<feature_count>},
    %features: i32 {swage.role = #swage.role<feature_count>}) {
  return
}
