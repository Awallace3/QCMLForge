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
test("final v5 export carries the matched-control benchmark slices",options,()=>{
  const e=data.evaluation;
  assert.equal(data.pendingModels.length,0);
  assert.equal(data.models["v5-axial"].sha256,
    "0a0eeee1f5f63d07b1db5940c0a55db12626627ab1df8cd723265ec876f9698b");
  assert.equal(data.models["v5-c2v"].sha256,
    "d1de460178ea5f8c03204a03bf7bb13d280b6340b60be5abe841038139329c73");
  near(e.corpusTest["v5-axial"].mae,.561702,1e-6);
  near(e.corpusTest["v5-c2v"].mae,.553118,1e-6);
  near(e.overall["v5-axial"].mae,.5185179355758134);
  near(e.overall["v5-c2v"].mae,.5490517215792757);
  near(e.pairDisjoint["v5-axial"].mae,.5103052978423416);
  near(e.pairDisjoint["v5-c2v"].mae,.5537329228926795);
  assert.equal(e.byClass["pi-pi stack"].n,48);
  near(e.byClass["pi-pi stack"]["v5-axial"].mae,.4997442967065658);
  near(e.byClass["pi-pi stack"]["v5-c2v"].mae,.6715250320065723);
  assert.deepEqual(Object.keys(e.byScale),["0.90","0.95","1.00","1.05","1.10","1.25","1.50","2.00"]);
  for(const scale of ["0.90","0.95","1.00"])
    assert.ok(e.byScale[scale]["v5-c2v"].mae>e.byScale[scale]["v5-axial"].mae);
  for(const scale of ["1.05","1.10","1.25","1.50"])
    assert.ok(e.byScale[scale]["v5-c2v"].mae<e.byScale[scale]["v5-axial"].mae);
});
test("benzene acid test and shared l=2 cap are exported without trajectory claims",options,()=>{
  const acid=data.evaluation.benzeneAcidTest.arms;
  near(acid["v4_axial"].pi_minus_inplane_perp,-2.8575598757174525e-4);
  near(acid["v5_axial"].pi_minus_inplane_perp,-3.4924198743402357e-4);
  near(acid["v5_c2v"].pi_minus_inplane_perp,.5173302671998468);
  near(acid["v5_c2v"].carbon.l2_norm_over_rho,.8661777336813905);
  near(Math.hypot(acid["v5_c2v"].carbon.a20,acid["v5_c2v"].carbon.a22c),
    acid["v5_c2v"].carbon.l2_norm);
  assert.equal(data.evaluation.telemetry.available,false);
  assert.match(data.evaluation.telemetry.reason,/no per-channel gradient norms/i);
  assert.match(data.evaluation.modelRelations.v5Control,/v5-axial/);
  assert.match(data.evaluation.modelRelations.v4ToV5,/typing/i);
});
test("browser surface polynomial reproduces the final benzene acid test",options,()=>{
  const d=data.dimers.find(d=>d.id===24),record=d.records["1.0"];
  const center=data.evaluation.benzeneAcidTest.benzene.center_atom;
  const directions=data.evaluation.benzeneAcidTest.benzene.probe_directions;
  for(const [exportKey,acidKey] of [
    ["v4-axial","v4_axial"],["v5-axial","v5_axial"],["v5-c2v","v5_c2v"]
  ]){
    const params=data.models[exportKey].parameters[record.types[center]];
    const expected=data.evaluation.benzeneAcidTest.arms[acidKey].benzene_angular_factor;
    for(const [direction,u] of Object.entries(directions))
      near(M.surface(params,record.poly[center],u),expected[direction],2e-13);
  }
});
