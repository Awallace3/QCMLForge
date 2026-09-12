/* Display-side algebra only. Authoritative pair energies come from Python. */
const ModelMath = (() => {
  function monomials([x,y,z]) {return [x,y,z,x*x,y*y,z*z,x*y,x*z,y*z];}
  const dot = (a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
  function harmonics(poly, direction) {
    const norm=Math.hypot(...direction);
    if (!Number.isFinite(norm) || norm<=0) throw Error("Invalid pair direction");
    const basis=monomials(direction.map(v=>v/norm));
    return poly.map(channel=>dot(channel,basis));
  }
  const angular = (parameters,h)=>1+dot(parameters.slice(2),h);
  function contribution(parameters,h,channel) {
    const slot=[0,3,6].indexOf(channel);
    return slot<0?0:parameters[slot+2]*h[slot];
  }
  function pair(dimer, record, positions, model, i, j) {
    const pa=model.parameters[record.types[i]], pb=model.parameters[record.types[j]];
    const delta=positions[j].map((v,k)=>v-positions[i][k]), r=Math.hypot(...delta);
    const hi=harmonics(record.poly[i],delta), hj=harmonics(record.poly[j],delta.map(v=>-v));
    const fa=angular(pa,hi), fb=angular(pb,hj), x=Math.sqrt(pa[1]*pb[1])*r;
    const shape=(1+x+x*x/3)*Math.exp(-x), radial=pa[0]*pb[0]*shape;
    return {pa,pb,hi,hj,fa,fb,x,r,shape,radial,energy:radial*fa*fb,
      index:i*(dimer.symbols.length-dimer.split)+(j-dimer.split)};
  }
  function surface(parameters, poly, direction) {
    return angular(parameters,harmonics(poly,direction));
  }
  function effectiveMode(model,type) {
    return model.arm==="shared-iso"?"isotropic":model.channels===2&&type.mode==="c2v"?"axial":type.mode;
  }
  return {monomials,harmonics,angular,contribution,pair,surface,effectiveMode};
})();
if (typeof module !== "undefined") module.exports=ModelMath;
