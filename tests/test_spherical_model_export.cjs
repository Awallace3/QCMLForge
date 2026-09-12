// MASTIFF_MODEL_DATA=/path/to/verified-export.json node --test tests/test_spherical_model_export.cjs
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const M=require("../scripts/spherical_vis/model_math.js");
const H=require("../scripts/spherical_vis/harmonics.js");
const path=process.env.MASTIFF_MODEL_DATA;
const data=path?JSON.parse(fs.readFileSync(path,"utf8")):null;
const options={skip:!path&&"Set MASTIFF_MODEL_DATA to test a real experiment export"};
function near(a,b,tolerance=2e-11){assert.ok(Math.abs(a-b)<=tolerance*Math.max(1,Math.abs(b)),`${a} != ${b}`);}
test("all 528 pair decompositions reproduce authoritative exported energies",options,()=>{
  assert.equal(data.schema,"mastiff-spherical-models-v1");
  let records=0,disjoint=0,pairs=0;
  for(const d of data.dimers)for(const [scale,record] of Object.entries(d.records)){
    records++;disjoint+=!record.sharedPair;
    for(const [key,model] of Object.entries(data.models)){
      let sum=0,minFactor=Infinity,negative=0;
      for(let i=0;i<d.split;i++)for(let j=d.split;j<d.symbols.length;j++){
        const p=M.pair(d,record,d.geometries[scale],model,i,j);
        near(p.energy,record.models[key].pairs[p.index]);
        sum+=p.energy;minFactor=Math.min(minFactor,p.fa,p.fb);
        negative+=p.energy<0;pairs++;
      }
      near(sum,record.models[key].total);
      near(minFactor,record.models[key].minAngular);
      assert.equal(negative,record.models[key].negativePairs);
      assert.ok(minFactor>=.2-1e-12);
    }
  }
  assert.equal(records,528);assert.equal(disjoint,416);
  assert.ok(pairs>100000);
});
test("real benzene carbon: v4 pi-face equals in-plane perpendicular",options,()=>{
  const d=data.dimers.find(d=>d.id===24),r=d.records["1.0"],positions=d.geometries["1.0"];
  const i=d.symbols.findIndex(s=>s==="C");
  const refs=r.plans[i].refs[0];
  assert.equal(d.symbols[refs[0]],"H","Authoritative benzene z must point C→H");
  const frame=H.frame(positions,i,refs);
  assert.ok(frame.x,"C2v geometry export must preserve transverse references");
  for(const key of ["v4-iso","v4-axial"]){
    const params=data.models[key].parameters[r.types[i]];
    near(M.surface(params,r.poly[i],frame.x),M.surface(params,r.poly[i],frame.y),1e-13);
    near(M.surface(params,r.poly[i],frame.x),1-params[3]/2,1e-13);
    assert.equal(params[4],0);
  }
});
test("all physical parameters obey positive radial and per-degree coefficient bounds",options,()=>{
  for(const model of Object.values(data.models))for(const p of Object.values(model.parameters)){
    assert.ok(p.every(Number.isFinite));
    assert.ok(p[0]>0&&p[1]>0);
    assert.ok(Math.abs(p[2])<.4);
    assert.ok(Math.hypot(p[3],p[4])<.4);
    if(model.channels===2)assert.equal(p[4],0);
    if(model.arm==="shared-iso")assert.deepEqual(p.slice(2),[0,0,0]);
  }
});
