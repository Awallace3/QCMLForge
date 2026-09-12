/* UI and 3Dmol rendering; all data are embedded in the generated HTML. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const H = Harmonics;
  const data = JSON.parse($("dataset").textContent);
  const originalPage = document.documentElement.outerHTML;
  const learned = Boolean(data.models);
  const distances = ["0.9","0.95","1.0","1.05","1.1","1.25","1.5","2.0"];
  const formulas = ["z","x","y","(3z² − 1)/2","√3 xz","√3 yz","√3 (x² − y²)/2","√3 xy"];
  let viewer, current = data.dimers[0], positions, selected = new Set();
  let overlays = [], overlayLabels = [], referenceChoice = {};
  let inspector;
  const vec = p => ({x:p[0], y:p[1], z:p[2]});
  const group = i => i < current.split ? "A" : "B";
  const atomName = i => `${group(i)}${i < current.split ? i+1 : i-current.split+1} ${current.symbols[i]}`;
  const radius = i => current.symbols[i] === "H" ? .22 : .32;
  function fail(error) {
    $("error").hidden = false;
    $("error").textContent = `Viewer error: ${error.message}. Check CDN access and WebGL availability.`;
    console.error(error);
  }
  function guarded(fn) {
    return (...args) => {try {return fn(...args);} catch (error) {fail(error);}};
  }
  function appendMesh(target, mesh) {
    const offset = target.vertexArr.length;
    target.vertexArr.push(...mesh.vertexArr);
    target.faceArr.push(...mesh.faceArr.map(i => i+offset));
    target.normalArr.push(...mesh.normalArr);
  }
  function outlineSphere(center, r) {
    const mesh=H.radialMesh(center,()=>r,16);
    mesh.normalArr=mesh.vertexArr.map(v=>vec(H.unit(H.sub(center,[v.x,v.y,v.z]))));
    return mesh;
  }
  function cylinder(a, b, r) {
    const z = H.unit(H.sub(b,a));
    const least = z.map(Math.abs).indexOf(Math.min(...z.map(Math.abs)));
    const gauge = [0,0,0]; gauge[least] = 1;
    const x = H.unit(H.cross(z,gauge)), y = H.cross(z,x);
    const vertexArr = [], faceArr = [], normalArr = [], n = 20;
    for (const p of [a,b]) {
      for (let j=0;j<n;j++) {
        const angle = 2*Math.PI*j/n;
        vertexArr.push(vec(p.map((v,k) => v+r*(x[k]*Math.cos(angle)+y[k]*Math.sin(angle)))));
        normalArr.push(vec(x.map((v,k)=>-v*Math.cos(angle)-y[k]*Math.sin(angle))));
      }
    }
    for (let j=0;j<n;j++) {
      const k=(j+1)%n;
      faceArr.push(j,k,j+n,k,k+n,j+n);
    }
    return {vertexArr,faceArr,normalArr};
  }
  function outlines() {
    // Back-facing expanded hulls produce blue/red silhouette rims while
    // preserving element-colored molecular interiors and normal depth testing.
    for (const monomer of ["A","B"]) {
      const mesh = {vertexArr:[],faceArr:[],normalArr:[]};
      current.symbols.forEach((_,i) => {
        if (group(i) === monomer) appendMesh(mesh,outlineSphere(positions[i],radius(i)+.045));
      });
      for (const [a,b] of current.bonds) {
        if (group(a) === monomer) appendMesh(mesh,cylinder(positions[a],positions[b],.165));
      }
      // 3Dmol's public bundle does not export its numeric side constants.
      viewer.addCustom({...mesh,color:monomer==="A"?"#2563eb":"#dc2626",side:1});
    }
  }
  function atomButtons() {
    $("atoms").replaceChildren();
    current.symbols.forEach((_,i) => {
      const button = document.createElement("button");
      button.className = "atom";
      button.textContent = atomName(i);
      button.dataset.atom = i;
      button.setAttribute("aria-pressed", selected.has(i));
      button.style.borderBottomColor = group(i)==="A"?"#2563eb":"#dc2626";
      button.onclick = guarded(()=>toggle(i));
      $("atoms").append(button);
    });
  }
  function toggle(i) {
    if (selected.has(i)) selected.delete(i); else selected.add(i);
    renderHarmonics();
  }
  function rebuildMolecule(resetCamera) {
    const view = viewer.getView();
    viewer.removeAllModels(); viewer.removeAllShapes(); viewer.removeAllLabels();
    overlays=[]; overlayLabels=[];
    positions = current.geometries[$("distance").value];
    if (learned) current.plans=current.records[$("distance").value].plans;
    const atoms = current.symbols.map((elem,i) => ({
      elem,...vec(positions[i]),serial:i,index:i,chain:group(i),bonds:[],bondOrder:[]
    }));
    for (const [a,b] of current.bonds) {
      atoms[a].bonds.push(b); atoms[b].bonds.push(a);
      atoms[a].bondOrder.push(1); atoms[b].bondOrder.push(1);
    }
    const model = viewer.addModel();
    model.addAtoms(atoms);
    model.setStyle({}, {stick:{radius:.13},sphere:{radius:.32}});
    model.setStyle({elem:"H"}, {stick:{radius:.13},sphere:{radius:.22}});
    model.setClickable({},true,guarded(atom=>toggle(atom.index)));
    outlines();
    $("title").textContent = `${String(current.id).padStart(2,"0")} · ${current.name} · ${$("distance").value} Rₑ`;
    atomButtons();
    if (resetCamera) {viewer.zoomTo();viewer.zoom(1.3);} else viewer.setView(view);
    renderHarmonics();
    if (inspector) inspector.update(current,$("distance").value);
  }
  function addAxis(center, axis, name, color) {
    const end = center.map((v,k)=>v+axis[k]*.95);
    overlays.push(viewer.addArrow({start:vec(center),end:vec(end),radius:.017,
      radiusRatio:2.8,mid:.8,color,clickable:false}));
    overlayLabels.push(viewer.addLabel(name,{position:vec(end),fontSize:10,
      fontColor:color,showBackground:false,inFront:false}));
  }
  function renderHarmonics() {
    overlays.forEach(shape=>viewer.removeShape(shape));
    overlayLabels.forEach(label=>viewer.removeLabel(label));
    overlays=[]; overlayLabels=[];
    const factor = $("harmonic").value==="factor";
    const display=learned?$("harmonicDisplay").value:"used";
    const plain=learned&&!factor&&display==="plain";
    const channel = Number($("harmonic").value), average = $("mode").value==="average";
    const scale = Number($("scale").value), info = [], invisible = [];
    const record=learned?current.records[$("distance").value]:null;
    const primary=learned?$("modelPrimary").value:null;
    const secondary=learned?$("modelCompare").value:null;
    const treatment=learned&&!plain?$("surfaceMode").value:"single";
    if(learned) {
      $("surfaceMode").disabled=plain;
      $("harmonicDisplayNote").textContent=factor
        ? "Full factor: fᵢ = 1 + Σ aₗₘ C̄ₗₘ. Choose a harmonic display mode to inspect an individual channel."
        : plain
          ? "Plain geometric basis: all eight channels are available where a full frame exists, regardless of model usage. No transverse gauge is invented for axial sites. Equivalent-frame averaging can cancel channels; choose one reference to inspect individual lobes. Model surface comparison is disabled in plain mode."
          : display==="scaled"
            ? "Scaled contribution: aₗₘ × C̄ₗₘ, including the learned sign and magnitude. Inactive channels are zero. This is one term of fᵢ, without its constant 1. Energy plots are unchanged."
            : "Model-used basis: active channels only, without coefficient scaling. Unused channels are zero. Energy plots are unchanged.";
    }
    $("frameInfo").replaceChildren();
    for (const i of selected) {
      const plan = current.plans[i];
      const activeRefs=learned&&!plain&&data.models[primary].channels===2
        ? [...new Set(plan.refs.map(refs=>refs[0]))].map(z=>[z]) : plan.refs;
      const allFrames = activeRefs.map(refs=>({...H.frame(positions,i,refs),orbit:refs[0]}));
      const chosen = Math.min(referenceChoice[i] || 0,Math.max(0,activeRefs.length-1));
      const frames = average ? allFrames : allFrames.slice(chosen,chosen+1);
      function evaluateModel(u,key) {
        const model=data.models[key], parameters=model.parameters[record.types[i]];
        let h;
        if (average) h=ModelMath.harmonics(record.poly[i],u);
        else {
          const refs=model.channels===2?activeRefs[chosen]:
            (data.models[primary].channels===2
              ?plan.refs.find(ref=>ref[0]===activeRefs[chosen]?.[0]):plan.refs[chosen]);
          const f=refs?H.frame(positions,i,model.channels===2?refs.slice(0,1):refs):null;
          h=[0,3,6].map(slot=>f?H.value(u,[f],slot,false):0);
        }
        if (factor) return ModelMath.angular(parameters,h);
        const slot=[0,3,6].indexOf(channel);
        if (slot<0 || slot>=model.channels || model.arm==="shared-iso") return 0;
        return display==="scaled"?ModelMath.contribution(parameters,h,channel):h[slot];
      }
      const fields=learned&&treatment==="overlay"?[primary,secondary]:[primary];
      let maximum=0;
      for (const key of fields) {
      const evaluate = u => plain ? H.plainValue(u,frames,channel,average) : learned
        ? evaluateModel(u,key)-(treatment==="difference"?evaluateModel(u,secondary):0)
        : H.value(u,frames,channel,average);
      for (const sign of [1,-1]) {
        let maxSign=0;
        const mesh = H.radialMesh(positions[i],u=>{
          const r = Math.max(0,sign*evaluate(u));
          maxSign=Math.max(maxSign,r); return scale*r;
        });
        maximum=Math.max(maximum,maxSign);
        const nullSurface=learned&&factor&&data.models[key].arm==="shared-iso"&&treatment!=="difference";
        const comparison=learned&&treatment==="overlay"&&key===secondary;
        if (maxSign>1e-6) overlays.push(viewer.addCustom({...mesh,
          color:nullSurface?"#93939b":comparison?"#ae6ad1":sign===1?"#079b87":"#c97913",
          opacity:treatment==="overlay"?.42:.68,side:2,clickable:false}));
      }
      }
      if (maximum<1e-6) invisible.push(atomName(i));
      if ($("axes").checked) for (const f of frames) {
        addAxis(positions[i],f.z,"z","#34343e");
        if (f.x) {addAxis(positions[i],f.x,"x","#cc2777");addAxis(positions[i],f.y,"y","#7953ce");}
      }
      const row=document.createElement("div");
      const effective=plain?"plain geometry":learned?ModelMath.effectiveMode(data.models[primary],data.types[record.types[i]]):plan.kind;
      row.textContent=`${atomName(i)} · ${effective} · ${activeRefs.length} reference frame(s)`;
      if (!average && activeRefs.length>1) {
        const select=document.createElement("select");
        select.className="select select-xs";
        select.style.width="auto"; select.style.marginLeft="8px";
        select.setAttribute("aria-label",`Reference frame for ${atomName(i)}`);
        activeRefs.forEach((refs,j)=>select.add(new Option(
          `z → ${atomName(refs[0])}${refs.length>1?`, x → ${atomName(refs[1])}`:""}`,j)));
        select.value=chosen;
        select.onchange=guarded(()=>{referenceChoice[i]=Number(select.value);renderHarmonics();});
        row.append(select);
      } else {
        row.append(document.createTextNode(" · "+activeRefs.map(refs=>
          `z→${atomName(refs[0])}${refs.length>1?` x→${atomName(refs[1])}`:""}`).join(" / ")));
      }
      $("frameInfo").append(row);
      info.push(atomName(i));
    }
    if ($("labels").checked) current.symbols.forEach((_,i)=>{
      overlayLabels.push(viewer.addLabel(atomName(i),{position:vec(positions[i]),fontSize:11,
        fontColor:group(i)==="A"?"#1747b3":"#b51c1c",backgroundColor:"#ffffff",backgroundOpacity:.8}));
    });
    document.querySelectorAll("[data-atom]").forEach(b=>b.setAttribute("aria-pressed",selected.has(Number(b.dataset.atom))));
    const allSelected = selected.size === current.symbols.length;
    $("toggleAll").textContent = allSelected ? "Hide all atoms’ lobes" : "Show all atoms’ lobes";
    $("toggleAll").setAttribute("aria-pressed", allSelected);
    $("selection").textContent=info.length
      ? `${info.length} atom(s) selected · ${factor?"learned fᵢ":`${plain?"plain":display==="scaled"&&learned?"scaled aₗₘ ×":"model-used"} C${H.labels[channel]}`}${learned&&!plain?` · ${treatment==="difference"?"model 1 − model 2":treatment==="overlay"?"model 1 teal / model 2 purple; isotropic grey":primary}`:""}${invisible.length?` · Zero / cancelled / unavailable gauge: ${invisible.join(", ")}`:""}`
      : "No atoms selected. Click an atom to reveal its harmonic; click again to hide it.";
    $("scaleValue").textContent=`${scale.toFixed(2)} Å`;
    viewer.render();
  }
  function listDimers() {
    const query=$("search").value.trim().toLowerCase();
    $("dimers").replaceChildren();
    const matches=data.dimers.filter(d=>`${d.id} ${d.name}`.toLowerCase().includes(query));
    $("count").textContent=`${matches.length} / 66 dimers`;
    for (const d of matches) {
      const button=document.createElement("button");
      button.className="dimer"; button.dataset.dimer=d.id;
      button.setAttribute("aria-current",d.id===current.id);
      button.textContent=`${String(d.id).padStart(2,"0")}  ${d.name}`;
      const category=document.createElement("small");
      category.textContent=d.id<=23?"Hydrogen-bonded":d.id<=46?"Dispersion-dominated":"Mixed";
      button.append(category);
      button.onclick=guarded(()=>{
        current=d;selected.clear();referenceChoice={};
        listDimers();rebuildMolecule(true);
      });
      $("dimers").append(button);
    }
  }
  guarded(()=>{
    if (typeof $3Dmol==="undefined") throw Error("3Dmol.js did not load");
    viewer=$3Dmol.createViewer($("viewer"),{backgroundColor:"#f5f3ef",antialias:true});
    viewer.setProjection("orthographic");
    distances.forEach(d=>$("distance").add(new Option(d,d)));
    $("distance").value="1.0";
    H.labels.forEach((label,i)=>$("harmonic").add(new Option(`C${label} · ${formulas[i]}`,i)));
    $("harmonic").value="3";
    if (learned) {
      $("modelControls").hidden=false;
      $("pairInspector").hidden=false;
      $("harmonicDisplayControl").hidden=false;
      $("harmonicDisplayNote").hidden=false;
      $("basisProvenance").hidden=true;
      $("harmonic").add(new Option("Learned angular factor fᵢ","factor"),0);
      $("harmonic").value="factor";
      $("harmonicDisplay").onchange=guarded(()=>{
        if($("harmonic").value==="factor")$("harmonic").value="3";
        renderHarmonics();
      });
      inspector=new ModelInspector(data,guarded(()=>{
        renderHarmonics();inspector.update(current,$("distance").value);
      }),guarded(scale=>{
        $("distance").value=scale;rebuildMolecule(false);
      }));
      $("modelImport").onchange=guarded(async event=>{
        try {
          const file=event.target.files[0];if(!file)return;
          const payload=JSON.parse(await file.text());
          if(payload.schema!=="mastiff-spherical-models-v1"||payload.dimers?.length!==66||!payload.models)
            throw Error("Not a complete verified model export");
          const page=new DOMParser().parseFromString(originalPage,"text/html");
          page.getElementById("dataset").textContent=JSON.stringify(payload).replaceAll("<","\\u003c");
          // Navigate to a fresh realm; document.write would redeclare the
          // existing global const bindings for Harmonics and ModelMath.
          location.replace(URL.createObjectURL(new Blob(
            ["<!doctype html>"+page.documentElement.outerHTML],{type:"text/html"}
          )));
        } catch(error) {fail(error);}
      });
    }
    $("sha").textContent=data.source_sha256;
    $("search").oninput=listDimers;
    $("distance").onchange=guarded(()=>rebuildMolecule(false));
    for (const id of ["harmonic","mode","axes","labels"]) $(id).onchange=guarded(renderHarmonics);
    $("scale").oninput=guarded(renderHarmonics);
    $("clear").onclick=guarded(()=>{selected.clear();renderHarmonics();});
    $("toggleAll").onclick=guarded(()=>{
      selected = selected.size === current.symbols.length
        ? new Set() : new Set(current.symbols.map((_, i)=>i));
      renderHarmonics();
    });
    $("reset").onclick=()=>{viewer.zoomTo();viewer.zoom(1.3);viewer.render();};
    new ResizeObserver(()=>{viewer.resize();viewer.render();}).observe($("stage"));
    listDimers();rebuildMolecule(true);
  })();
})();
