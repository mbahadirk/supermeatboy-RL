// Native adapter control-flow tests with fake memory only. No game/save is accessed.
const vm = require('node:vm');
const assert = require('node:assert/strict');
const load = require('./native_harness.cjs');
const {context, sent, replacements, hooks, Mem, nativeCalls} = load({realtest:true, finish_quota:5});
context.goPayload={level_name:'4-1',realtest_par:10};
vm.runInContext(`
  curName='unknown';wave=0;
  waitGo();
  assert.equal(curName,'4-1');assert.equal(realtestPar,10);
`,context);
context.goPayload={level_name:'4-2',realtest_par:10};
vm.runInContext(`waitGo();assert.equal(curName,'4-1');`,context);
vm.runInContext(`
  curName='4-1'; wave=3; frame=65; ready=true; main=new Mem(0x200000);
  hud=new Mem(0x210000); hud.add(0x384).writeFloat(9.25);
  agents=[new Mem(0x220000)]; status=[2];finishFrame=[64];finishSeconds=[9.25];realtestPar=10;
  assert.equal(nativeFinishSeconds(),9.25);
  rememberFinish(0);
  const result={level:'4-1',wave:3,agent:0,seconds:9.25,first_wave:3,after_wave:8};
  assert.equal(finishFromWave(result),false); // five additional waves are mandatory
  wave=8;
  assert.equal(finishFromWave({...result,wave:2}),false);
  assert.equal(finishFromWave({...result,level:'4-2'}),false);
  assert.equal(finishFromWave({...result,seconds:9.5}),false);
  assert.equal(finishFromWave({...result,agent:2}),false);
  finishSeconds[0]=11;
  assert.equal(finishFromWave({...result,seconds:11}),false);
  finishSeconds[0]=9.25;
`,context);
context.goPayload={agents:1,realtest_par:10,finish:{level:'4-1',wave:3,agent:0,seconds:9.25,first_wave:3,after_wave:8}};
vm.runInContext(`
  status=[1];finishSeconds=[null];finishFrame=[-1];
  endWave('done');
  assert.equal(finishRequested,true);assert.equal(finishing,60);assert.equal(finishTime,9.25);
  assert.equal(frame,65);assert.equal(pendingWave,false); // no reset may overwrite the finishing clock
  assert.equal(finishFromWave({level:'4-1',wave:3,agent:0,seconds:9.25}),false);
`,context);
assert.equal(sent.find(m=>m.ev==='wave_end').finish_seconds[0],null); // final wave may have no winner
// The actual native-finish update writes this attempt's time, not the historical best or999.99.
vm.runInContext(`goalPieces=[[10,20]];`,context);
replacements.get(0x519d40)(new Mem(0x200000));
vm.runInContext(`assert.equal(hud.add(0x384).readFloat(),9.25);`,context);

// After the snap animation, the native transition keeps ownership. No repeated PPO wave_end.
vm.runInContext(`finishing=0;`,context);
const terminalCount=sent.filter(m=>m.ev==='wave_end').length;
for(let n=0;n<100;n++) replacements.get(0x519d40)(new Mem(0x200000));
assert.equal(sent.filter(m=>m.ev==='wave_end').length,terminalCount);
vm.runInContext(`assert.equal(frame,65);`,context);
// Native result-screen continuation works without polling XInput or an attached controller.
vm.runInContext(`hud.add(0x38c).writeU32(13);`,context);
function hudTick(){
 const receiver={context:{ecx:new Mem(0x210000)}};
 hooks.get(0x4a1810).onEnter.call(receiver);
 hooks.get(0x4a1810).onLeave.call(receiver);
}
for(let n=0;n<59;n++)hudTick();
assert.equal(nativeCalls.filter(c=>c.address===0x4aa050).length,0);
hudTick();
assert.equal(nativeCalls.filter(c=>c.address===0x4aa050).length,1);
assert.equal(nativeCalls.find(c=>c.address===0x4aa050).options.abi,'mscdecl');
assert.equal(nativeCalls.find(c=>c.address===0x4aa050).options.exceptions,'propagate');
assert.equal(sent.filter(m=>m.ev==='continue').length,1);
for(let n=0;n<180;n++)hudTick();
assert.equal(nativeCalls.filter(c=>c.address===0x4aa050).length,1); // no duplicate transition
vm.runInContext(`assert.equal(continueRequested,true);`,context);

// The swallowed agent-goal hook captures native HUD time at the successful contact.
vm.runInContext(`
  finishRequested=false;finishing=0;status=[0];finishFrame=[-1];finishSeconds=[null];
  curAgent=0;frame=12;hud.add(0x384).writeFloat(8.75);
  agents[0].add(0x84).writeFloat(10);agents[0].add(0x88).writeFloat(20);
`,context);
replacements.get(0x4c1910)(new Mem(0),0);
vm.runInContext(`assert.equal(status[0],2);assert.equal(finishFrame[0],12);assert.equal(finishSeconds[0],8.75);`,context);

// Without the flag, S remains a manual forced finish excluded from player timing records.
const old = load();
vm.runInContext(`
  ready=true;main=new Mem(0x200000);hud=new Mem(0x210000);
  assert.equal(finishFromWave({level:'4-1',wave:1,agent:0,seconds:5}),false);
  assert.equal(requestFinish(null),true);assert.equal(finishTime,999.99);
  assert.equal(requestFinish(5),false);
`,old.context);
console.log('PPO realtest native clock, checked directive, finish and duplicate guards passed.');

// New level clears the old grade/countdown, finish ownership, and outstanding menu input.
hooks.get(0x4e9020).onEnter([{readPointer:()=>({readCString:()=> '4-2'})}]);
vm.runInContext(`
 assert.equal(curName,'4-2');assert.equal(wave,0);assert.equal(finishRequested,false);
 assert.equal(realtestBest,null);assert.equal(realtestFirstWave,null);
 assert.equal(completeFrames,0);assert.equal(continueRequested,false);
`,context);
