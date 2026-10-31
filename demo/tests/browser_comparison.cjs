// Browser QA uses fixture-only model transport; no external inference calls.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const fs=require('node:fs');
fs.mkdirSync('runs/ui-comparison-qa',{recursive:true});
let server;
async function fixtureServer(){
 server=spawn(process.env.PYTHON || 'python3',['-m','demo.tests.comparison_fixture_server'],{stdio:['ignore','pipe','pipe']});
 return new Promise((resolve,reject)=>{let output='';const timer=setTimeout(()=>reject(Error('Fixture server did not start')),15000);server.stdout.on('data',chunk=>{output+=chunk;const line=output.split('\n').find(s=>s.startsWith('http://'));if(line){clearTimeout(timer);resolve(line);}});server.stderr.on('data',chunk=>process.stderr.write(chunk));server.once('error',reject);});
}
(async()=>{
 const base=await fixtureServer(),browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 const errors=[];
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}});page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/compare');await page.waitForFunction(()=>!document.querySelector('#start-run').disabled);
  const catalog=await (await page.request.get(base+'/api/comparison-cases')).json();assert.ok(catalog.length);const caseId=catalog[0].id;
  await page.selectOption('#model','fixture-model');await page.selectOption('#scenario',caseId);assert.equal(await page.locator('#variant').inputValue(),'seeds_select');
  await page.locator('#start-run').click();await page.waitForFunction(()=>document.querySelector('#run-status').textContent.startsWith('completed'));
  const id=await page.evaluate(()=>sessionStorage.getItem('comparison-job'));const state=await (await page.request.get(base+'/api/comparisons/'+id)).json();
  assert.equal(state.mode,'published_attribution');assert.equal(state.variant,'seeds_select');
  assert.equal(state.events.some(e=>e.kind.startsWith('repair_')||e.kind==='verification_completed'),false);
  assert.equal(await page.locator('.stage.active').count(),0,'Finished systems must not retain active algorithm stages');
  for(const side of ['agentrx','agentreroute']){
   assert.match(await page.locator('#'+side+'-prediction').textContent(),new RegExp('step '+state.results[side].prediction.step_index));
   assert.match(await page.locator('#'+side+'-outcome').textContent(),/Exact step match|Step mismatch/);
  }
  assert.match(await page.locator('#agentreroute-candidates').textContent(),/policy weight|No ranked/);
  const requests=state.events.filter(e=>e.system==='agentrx'&&e.kind==='request_started');assert.ok(requests.length>=2);
  const cursor=state.events.findIndex(e=>e.seq===requests[1].seq)+1;
  await page.locator('#cursor').evaluate((el,value)=>{el.value=value;el.dispatchEvent(new Event('input'));},cursor);
  await page.locator('#cursor').evaluate(el=>el.dispatchEvent(new Event('input')));
  assert.equal(await page.locator('#agentrx-calls').textContent(),'2','Historical request counter must be stable across rerenders');
  await page.locator('#cursor').evaluate(el=>{el.value=0;el.dispatchEvent(new Event('input'));});
  assert.match(await page.locator('#agentrx-candidates').textContent(),/Waiting for attribution/);
  assert.match(await page.locator('#agentreroute-outcome').textContent(),/Waiting for attribution/);
  assert.doesNotMatch(await page.locator('#agentreroute-prediction').textContent(),/Gold/);
  assert.equal(await page.locator('#trace button').count(),0,'Zero cursor must not reveal later input');
  await page.locator('#go-live').click();await page.reload();await page.waitForFunction(()=>document.querySelector('#run-status').textContent.startsWith('completed'));
  assert.equal(await page.locator('#model').inputValue(),'fixture-model');assert.equal(await page.locator('#scenario').inputValue(),caseId);
  assert.equal(await page.locator('#agentreroute-evidence').evaluate(el=>el.open),false);
  await page.locator('#agentrx-evidence > summary').click();await page.locator('#agentrx-log button').first().click();
  assert.equal(await page.locator('#event-inspector').evaluate(el=>el.open),true);assert.match(await page.locator('#event-detail').textContent(),/request_completed/);
  await page.locator('#go-live').click();await page.locator('#agentrx-evidence > summary').click();await page.locator('#event-inspector > summary').click();
  await page.locator('.input-run > summary').click();await page.locator('#trace button').first().click();assert.match(await page.locator('#trace-detail').textContent(),/"index"|"step_index"/);
  await page.screenshot({path:'runs/ui-comparison-qa/compare-fixture-desktop.png',fullPage:true});await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:'runs/ui-comparison-qa/compare-fixture-mobile.png',fullPage:true});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile document overflows');
  await page.emulateMedia({reducedMotion:'reduce'});await page.setViewportSize({width:1440,height:1000});
  await page.locator('#start-run').click();await page.locator('#cancel-run').click();await page.waitForFunction(()=>document.querySelector('#run-status').textContent.startsWith('canceled'));
  assert.equal(await page.locator('#agentreroute-status').textContent(),'Not run');
  await page.selectOption('#model','invalid-synthesis');await page.selectOption('#variant','synth_select_p10');await page.locator('#start-run').click();
  await page.waitForFunction(()=>document.querySelector('#run-status').textContent.startsWith('partial'));
  assert.equal(await page.locator('#agentreroute-status').textContent(),'partial');assert.match(await page.locator('#agentreroute-checks').textContent(),/Seed-only fallback/);
  assert.ok(await page.locator('#agentreroute .stage.failed').count()>0);assert.match(await page.locator('#agentreroute-warning').textContent(),/Seed-only fallback/);
  assert.equal(await page.locator('#start-run').isEnabled(),true,'Partial is terminal');
  // A trained checkpoint is a distinct frozen setting, retained in saved runs.
  await page.selectOption('#model','fixture-model');await page.selectOption('#variant','seeds_select');
  await page.selectOption('#policy-mode','trained');await page.locator('#start-run').click();
  await page.waitForFunction(()=>document.querySelector('#run-status').textContent.startsWith('completed'));
  const trainedId=await page.evaluate(()=>sessionStorage.getItem('comparison-job'));
  const trained=await (await page.request.get(base+'/api/comparisons/'+trainedId)).json();
  assert.equal(trained.policy_mode,'trained');assert.equal(trained.results.agentreroute.policy_mode,'trained');
  assert.match(await page.locator('#record-context').textContent(),/trained checkpoint/);
  await page.reload();await page.waitForFunction(()=>document.querySelector('#run-status').textContent.startsWith('completed'));
  assert.equal(await page.locator('#policy-mode').inputValue(),'trained');
  // A full-trace decision may differ from the highest-policy-weight candidate.
  const outside=await browser.newPage();outside.on('pageerror',e=>errors.push(e.message));
  const custom=structuredClone(state);custom.id='b'.repeat(32);custom.results.agentreroute.prediction={agent:'assistant',step_index:999,category:'fixture',rationale:'Outside-pool test answer'};
  for(const event of custom.events){if(event.system!=='agentreroute')continue;if(event.kind==='attribution_completed'||event.kind==='system_finished'){const target=event.result||event;target.prediction=custom.results.agentreroute.prediction;target.selection={status:'ok',in_pool:false,fell_back:false};}}
  await outside.route('**/api/comparisons/'+custom.id+'*',route=>route.fulfill({json:custom}));await outside.goto(base+'/compare?run='+custom.id);
  await outside.locator('#agentreroute-prediction').filter({hasText:'step 999'}).waitFor();assert.match(await outside.locator('#agentreroute-prediction').textContent(),/outside the evidence shortlist/);assert.doesNotMatch(await outside.locator('#agentreroute-candidates').textContent(),/step 999/);
  await outside.close();
  // Legacy saved runs remain readable even when the current published corpus is absent.
  const legacy=await browser.newPage();legacy.on('pageerror',e=>errors.push(e.message));
  const oldId='a'.repeat(32),verification={task_outcome:{after:{reward:1,state_ok:true,missing_outputs:[]}},verification:{accepted:true},decision:{status:'accepted'},policy_update:{reward:1}};
  const old={id:oldId,scenario:'historical-retail',model:'historical-model',provider:'ollama',legacy:true,mode:'legacy_recovery',variant:'legacy',status:'completed',artifact_dir:'saved-history',events:[{seq:1,system:'agentreroute',kind:'verification_completed',...verification},{seq:2,system:'agentreroute',kind:'system_finished',result:{status:'completed',metrics:{}}},{seq:3,kind:'job_finished',status:'completed'}]};
  await legacy.route('**/api/comparison-cases',route=>route.fulfill({json:[]}));await legacy.route('**/api/comparisons/'+oldId+'*',route=>route.fulfill({json:old}));
  await legacy.goto(base+'/compare?run='+oldId);await legacy.locator('#agentreroute-outcome').filter({hasText:'Historical recovery'}).waitFor();
  assert.match(await legacy.locator('#record-context').textContent(),/Historical legacy recovery/);assert.match(await legacy.locator('#agentreroute-outcome').textContent(),/Historical policy update/);
  assert.equal(await legacy.locator('#model').inputValue(),'historical-model');assert.equal(await legacy.locator('#scenario').inputValue(),'historical-retail');assert.equal(await legacy.locator('#start-run').isDisabled(),true);
  assert.match(await legacy.locator('#error').textContent(),/Published benchmark cases are unavailable/);await legacy.close();
  assert.deepEqual(errors,[]);console.log('PASS: attribution, exact scoring, outside-pool answer, variants, partial fallback, legacy playback, counters, mobile, saved URLs, cancellation; fixture transport only');
 }finally{await browser.close();server.kill('SIGTERM');}
})().catch(e=>{console.error(e.stack);if(server)server.kill('SIGTERM');process.exitCode=1;});
