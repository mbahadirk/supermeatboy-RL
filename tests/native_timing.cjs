const assert=require('node:assert/strict');
const vm=require('node:vm');
const load=require('./native_harness.cjs');
const run=load({lock_step:true});
assert.equal(run.patches.length,1);
assert.equal(run.patches[0].address,0x450ef6);
assert.equal(run.heap.get(0x450ef6),0xeb);
assert.equal(run.context.rpc.exports.shutdown(),true);
assert.equal(run.heap.get(0x450ef6),0x7c);
assert.equal(run.patches.length,2);
assert.equal(run.context.rpc.exports.shutdown(),true); // cleanup is idempotent
assert.equal(run.patches.length,2);
const variable=load({lock_step:false});
assert.equal(variable.patches.length,0);
assert.throws(()=>load({lock_step:true,lockBytes:[0x90,0x90]}),/Cannot establish/);
assert.throws(()=>load({lock_step:true,lockBytes:[0xeb,0x53]}),/Cannot establish/); // not our lock
console.log('Fixed-step ownership, restoration, opt-out and byte validation passed.');
