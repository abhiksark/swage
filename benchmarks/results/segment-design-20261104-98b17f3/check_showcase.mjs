// benchmarks/results/segment-design-20261104-98b17f3/check_showcase.mjs
// Verify the local showcase with the installed browser, without dependencies.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, readFile, rm, writeFile, access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, dirname, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

const directory = dirname(fileURLToPath(import.meta.url));
assert(process.argv[2], 'Pass a new output directory as the first argument');
const destination = resolve(process.argv[2]);
await mkdir(destination, {recursive: false});
const profile = await mkdtemp(join(tmpdir(), 'swage-design-browser-'));
const browser = spawn(
  process.env.SWAGE_REVIEW_BROWSER || 'chromium',
  ['--headless', '--no-sandbox', '--disable-gpu', '--remote-debugging-port=0',
   `--user-data-dir=${profile}`, 'about:blank'],
  {stdio: 'ignore'},
);
let socket;
const pending = new Map();
const errors = [];
let sequence = 0;
async function command(method, params = {}) {
  const id = ++sequence;
  const promise = new Promise((accept, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`CDP timeout: ${method}`));
    }, 10000);
    pending.set(id, {accept, reject, timer});
  });
  socket.send(JSON.stringify({id, method, params}));
  return promise;
}
async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {
    expression, returnByValue: true, awaitPromise: true,
  });
  assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function screenshot(name) {
  const metrics = await command('Page.getLayoutMetrics');
  const size = metrics.cssContentSize;
  const result = await command('Page.captureScreenshot', {
    format: 'png', captureBeyondViewport: true,
    clip: {x: 0, y: 0, width: size.width, height: size.height, scale: 1},
  });
  await writeFile(join(destination, name), Buffer.from(result.data, 'base64'));
}
try {
  let port;
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8'))
        .split('\n')[0];
      break;
    } catch {
      await delay(50);
    }
  }
  assert(port, 'Browser did not start');
  const target = await (await fetch(
    `http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'},
  )).json();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((accept, reject) => {
    socket.addEventListener('open', accept, {once: true});
    socket.addEventListener('error', reject, {once: true});
  });
  socket.addEventListener('message', event => {
    const response = JSON.parse(event.data);
    if (response.method === 'Runtime.exceptionThrown') errors.push(response.params);
    if (!response.id) return;
    const waiter = pending.get(response.id);
    if (!waiter) return;
    pending.delete(response.id);
    clearTimeout(waiter.timer);
    if (response.error) waiter.reject(new Error(JSON.stringify(response.error)));
    else waiter.accept(response.result);
  });
  await command('Runtime.enable');
  await command('Page.enable');
  await command('Emulation.setDeviceMetricsOverride', {
    width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false,
  });
  await command('Page.navigate', {
    url: pathToFileURL(join(directory, 'showcase.html')).href,
  });
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await evaluate('document.readyState === "complete" && document.title.startsWith("Swage:")')) break;
    await delay(50);
  }
  assert.equal(await evaluate('document.title'),
    'Swage: choosing a useful programmable compiler');
  assert.deepEqual(await evaluate(
    "['route','program','chart'].map(x=>document.querySelectorAll('[data-'+x+']').length)"
  ), [3, 4, 3]);
  for (const key of ['a', 'b', 'c']) {
    await evaluate(`document.querySelector('[data-route="${key}"]').click()`);
    assert.equal(await evaluate(`document.querySelector('[data-route="${key}"]').getAttribute('aria-pressed')`), 'true');
    assert.equal(await evaluate("document.getElementById('route-title').textContent"),
      await evaluate(`routes.${key}.title`));
  }
  for (const key of ['energy', 'rational', 'normalize', 'weighted']) {
    await evaluate(`document.querySelector('[data-program="${key}"]').click()`);
    assert.equal(await evaluate("document.getElementById('program-code').textContent"),
      await evaluate(`programs.${key}.code`));
  }
  for (const [key, count] of [['fresh', 9], ['rows', 8], ['prepared', 6]]) {
    await evaluate(`document.querySelector('[data-chart="${key}"]').click()`);
    assert.equal(await evaluate("document.querySelectorAll('.chartrow').length"), count);
    assert.equal(await evaluate("document.querySelectorAll('.filters [aria-pressed=true]').length"), 1);
  }
  const links = await evaluate("Array.from(document.querySelectorAll('a[href]'), a=>a.getAttribute('href'))");
  let checkedLinks = 0;
  for (const href of links) {
    if (/^https?:/.test(href)) continue;
    if (href.startsWith('#')) {
      assert(await evaluate(`Boolean(document.getElementById(${JSON.stringify(href.slice(1))}))`));
    } else {
      await access(resolve(directory, href.split('#')[0]));
    }
    checkedLinks++;
  }
  await evaluate("showRoute('a');showProgram('energy');showChart('fresh')");
  assert(await evaluate('document.documentElement.scrollWidth <= innerWidth'));
  await screenshot('showcase-desktop.png');
  await command('Emulation.setDeviceMetricsOverride', {
    width: 390, height: 844, deviceScaleFactor: 1, mobile: false,
  });
  await delay(100);
  assert(await evaluate('document.documentElement.scrollWidth <= innerWidth'));
  await screenshot('showcase-mobile.png');
  assert.deepEqual(errors, []);
  const result = {
    passed: true, route_choices: 3, program_choices: 4, chart_cases: [9, 8, 6],
    local_links_checked: checkedLinks, javascript_errors: errors,
    viewports: [[1440, 1000], [390, 844]], horizontal_overflow: false,
    browser: (await command('Browser.getVersion')).product,
  };
  await writeFile(join(destination, 'showcase-check.json'),
    JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
} finally {
  for (const {timer} of pending.values()) clearTimeout(timer);
  socket?.close();
  browser.kill('SIGTERM');
  await new Promise(accept => browser.exitCode !== null
    ? accept() : browser.once('exit', accept));
  await rm(profile, {recursive: true, force: true});
}
