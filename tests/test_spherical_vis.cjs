// Run with: node --test tests/test_spherical_vis.cjs
const {test} = require("node:test");
const assert = require("node:assert/strict");
const H = require("../scripts/spherical_vis/harmonics.js");
const M = require("../scripts/spherical_vis/model_math.js");
const near = (a,b) => assert.ok(Math.abs(a-b) < 1e-12, `${a} != ${b}`);
test("positive Cartesian Racah convention and right-handed frame", () => {
  assert.deepEqual(H.basis([0,0,1]), [1,0,0,1,0,0,0,0]);
  near(H.basis([1,0,0])[6], Math.sqrt(3)/2);
  const f = H.frame([[0,0,0],[0,0,2],[2,0,1]],0,[1,2]);
  assert.deepEqual(f.x,[1,0,0]);
  assert.deepEqual(f.y,[0,1,0]);
  near(H.value([1,0,0],[f],1,true),1);
});
test("averaging suppresses sine channels and axial sites have no gauge", () => {
  const f = H.frame([[0,0,0],[0,0,1],[1,0,0]],0,[1,2]);
  near(H.value([0,1,0],[f],2,true),0);
  near(H.value([0,1,0],[f],2,false),1);
  for (const ch of [1,2,4,5,6,7]) near(H.value([1,2,3],[{z:[0,0,1]}],ch,false),0);
  near(H.value([0,0,1],[f,{z:[0,0,-1],x:[1,0,0],y:[0,-1,0]}],0,true),0);
});
test("degenerate frames fail and spherical meshes have finite valid indices", () => {
  assert.throws(() => H.frame([[0,0,0],[0,0,1],[0,0,2]],0,[1,2]), /Degenerate/);
  const mesh = H.radialMesh([1,2,3], u => Math.abs(H.basis(u)[3]), 12);
  assert.ok(mesh.vertexArr.every(v => Object.values(v).every(Number.isFinite)));
  assert.ok(mesh.faceArr.every(i => i >= 0 && i < mesh.vertexArr.length));
});
test("learned surfaces preserve axial degeneracy and the C22c channel", () => {
  // Lab z polar axis, x transverse axis. API polynomial ordering is documented.
  const poly=[[0,0,1,0,0,0,0,0,0],[0,0,0,-.5,-.5,1,0,0,0],
    [0,0,0,Math.sqrt(3)/2,-Math.sqrt(3)/2,0,0,0,0]];
  const axial=[1,2,.1,-.2,0], c2v=[1,2,.1,-.2,.15];
  near(M.surface(axial,poly,[1,0,0]),M.surface(axial,poly,[0,1,0]));
  near(M.surface(c2v,poly,[1,0,0])-M.surface(c2v,poly,[0,1,0]),Math.sqrt(3)*.15);
});
test("pair endpoints use opposite directions and keep computation unrounded", () => {
  const poly=[[0,0,1,0,0,0,0,0,0],[0,0,0,-.5,-.5,1,0,0,0],
    [0,0,0,0,0,0,0,0,0]];
  const p=M.pair({symbols:["H","H"],split:1},{types:["t","t"],poly:[poly,poly]},
    [[0,0,0],[0,0,2]],{parameters:{t:[2,1,.1,.2,0]}},0,1);
  near(p.fa,1.3);near(p.fb,1.1);near(p.x,2);
  near(p.energy,4*1.3*1.1*(1+2+4/3)*Math.exp(-2));
});
test("plain basis retains non-model channels and averages within axial orbits", () => {
  const f={x:[1,0,0],y:[0,1,0],z:[0,0,1],orbit:0};
  const g={x:[0,1,0],y:[-1,0,0],z:[0,0,1],orbit:0};
  const h={z:[1,0,0],orbit:1};
  near(H.plainValue([0,1,0],[f],2,false),1);
  near(H.plainValue([1,0,0],[f],6,true),Math.sqrt(3)/2);
  near(H.plainValue([0,0,1],[f,g,h],0,true),.5);
  near(H.plainValue([1,0,0],[h],6,false),0);
});
test("coefficient-scaled basis preserves amplitude and coefficient sign", () => {
  const p=[2,1,-.06,-.32,0],h=[.5,-.4,.8];
  near(M.contribution(p,h,0),-.03);
  near(M.contribution(p,h,3),.128);
  near(M.contribution(p,h,6),0);
  near(M.contribution(p,h,2),0);
  near(1+M.contribution(p,h,0)+M.contribution(p,h,3),M.angular(p,h));
});
