/* Real positive-Cartesian Racah basis, matching apnet_pt.mastiff. */
const Harmonics = (() => {
  const labels = ["10", "11c", "11s", "20", "21c", "21s", "22c", "22s"];
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const cross = (a, b) => [
    a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]
  ];
  function unit(a) {
    const norm = Math.hypot(...a);
    if (!Number.isFinite(norm) || norm < 1e-10) throw Error("Degenerate local frame");
    return a.map(v => v / norm);
  }
  function basis(v) {
    const [x, y, z] = unit(v), r = Math.sqrt(3);
    return [z, x, y, (3*z*z-1)/2, r*x*z, r*y*z, r*(x*x-y*y)/2, r*x*y];
  }
  function frame(positions, center, refs) {
    const z = unit(sub(positions[refs[0]], positions[center]));
    if (refs.length === 1) return {z};
    const raw = sub(positions[refs[1]], positions[center]);
    const x = unit(raw.map((v, i) => v - dot(raw, z)*z[i]));
    return {x, y: cross(z, x), z};
  }
  function value(direction, frames, channel, averaged) {
    if (!frames.length) return 0;
    const u = unit(direction);
    const selected = averaged ? frames : frames.slice(0, 1);
    return selected.reduce((sum, f) => {
      const z = dot(u, f.z);
      if (!f.x) return sum + (channel === 0 ? z : channel === 3 ? (3*z*z-1)/2 : 0);
      if (averaged && [2, 5, 7].includes(channel)) return sum;
      return sum + basis([dot(u, f.x), dot(u, f.y), z])[channel];
    }, 0) / selected.length;
  }
  function radialMesh(center, radius, resolution = 36) {
    const vertices = [], faces = [], cols = resolution * 2 + 1;
    for (let i = 0; i <= resolution; i++) {
      const theta = Math.PI * i / resolution;
      for (let j = 0; j < cols; j++) {
        const phi = 2*Math.PI*j/(cols-1);
        const u = [Math.sin(theta)*Math.cos(phi), Math.sin(theta)*Math.sin(phi), Math.cos(theta)];
        const r = radius(u);
        vertices.push({x: center[0]+r*u[0], y: center[1]+r*u[1], z: center[2]+r*u[2]});
        if (i < resolution && j < cols-1) {
          const a = i*cols+j, b = a+1, c = a+cols, d = c+1;
          faces.push(a, c, b, b, c, d);
        }
      }
    }
    return {vertexArr: vertices, faceArr: faces};
  }
  function plainValue(direction, frames, channel, averaged) {
    if (!frames.length) return 0;
    if (!averaged) return value(direction, frames.slice(0,1), channel, false);
    // Geometry-only inspection: no model channel or sine mask. Weight each
    // distinct z orbit equally, rather than counting tied transverse rows twice.
    const groups=new Map();
    for(const f of frames){
      const key=f.orbit;
      if(!groups.has(key))groups.set(key,[]);
      groups.get(key).push(f);
    }
    let total=0;
    for(const group of groups.values())
      total+=group.reduce((sum,f)=>sum+value(direction,[f],channel,false),0)/group.length;
    return total/groups.size;
  }
  return {labels, sub, dot, cross, unit, basis, frame, value, radialMesh, plainValue};
})();
if (typeof module !== "undefined") module.exports = Harmonics;
