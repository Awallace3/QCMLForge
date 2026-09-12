/* Trained-model diagnostics, separate from the basis/atom-toggle viewer. */
class ModelInspector {
  constructor(data, onModels, onScale) {
    this.data=data;this.onScale=onScale;this.dimer=null;this.scale="1.0";
    this.i=0;this.j=0;
    const $=id=>document.getElementById(id);
    this.$=$;
    for(const id of ["modelPrimary","modelCompare"]) {
      $(id).replaceChildren();
      for(const [key,m] of Object.entries(data.models))
        $(id).add(new Option(`${key} · step ${m.step}`,key));
      for(const key of data.pendingModels||[]) if(!data.models[key]) {
        const option=new Option(`${key} · checkpoint unavailable`,key);
        option.disabled=true;$(id).add(option);
      }
      $(id).onchange=onModels;
    }
    $("modelPrimary").value=data.models["v4-axial"]?"v4-axial":Object.keys(data.models)[0];
    $("modelCompare").value=data.models["v4-iso"]?"v4-iso":Object.keys(data.models)[0];
    $("surfaceMode").onchange=onModels;
    this.viewer=$3Dmol.createViewer($("pairViewer"),{backgroundColor:"#f5f3ef",antialias:true});
    this.viewer.setProjection("orthographic");
    for(const id of ["pairA","pairB"]) $(id).onchange=()=>{
      this.i=Number($("pairA").value);this.j=Number($("pairB").value);this.refreshPair(false);
    };
    $("pairReset").onclick=()=>{this.viewer.zoomTo();this.viewer.zoom(1.5);this.viewer.render();};
    $("allTypes").onchange=()=>this.parameters();
    new ResizeObserver(()=>{this.viewer.resize();this.viewer.render();}).observe($("pairViewer").parentElement);
    this.math($("generalEquation"),[
      String.raw`E_{ij}=A_i A_j\,f_i(\hat{\mathbf r}_{ij})\,f_j(-\hat{\mathbf r}_{ij})\left(1+x+\frac{x^2}{3}\right)e^{-x}`,
      String.raw`x=\sqrt{B_iB_j}\,r_{ij},\qquad f_i=1+\sum_{k\in\{10,20,22c\}}a_{ik}\,\overline C_{ik}`,
      String.raw`\overline C_{ik}=\frac{1}{|Z_i|}\sum_{z\in Z_i}\left(\frac{1}{|X_{iz}|}\sum_{x\in X_{iz}}C_k(F_{izx}^{T}\hat{\mathbf r}_{ij})\right)`
    ]);
    const note=document.createElement("p");note.className="subtle";
    note.textContent="For axial groups, C10/C20 need only z; absent transverse gauges contribute zero C22c. A has units √(kcal/mol), B is Å⁻¹, r is Å, and f and x are dimensionless. Ring-mode availability alone does not activate C22c in v4.";
    $("generalEquation").append(note);
    $("modelProvenance").textContent=JSON.stringify({
      dataset_sha256:data.source_sha256,...data.provenance,
      checkpoints:Object.fromEntries(Object.entries(data.models).map(([key,m])=>[key,{
        arm:m.arm,step:m.step,sha256:m.sha256,path:m.checkpoint
      }]))
    },null,2);
  }
  escape(s) {return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
  f(n) {return Number(n).toFixed(2);}
  name(i) {return `${i<this.dimer.split?"A":"B"}${i<this.dimer.split?i+1:i-this.dimer.split+1} ${this.dimer.symbols[i]}`;}
  keys() {return [...new Set([this.$("modelPrimary").value,this.$("modelCompare").value])];}
  math(element, equations) {
    element.replaceChildren();
    for(const equation of equations) {
      const node=document.createElement("div");node.className="equation";
      if(typeof katex!=="undefined") katex.render(equation,node,{displayMode:true,throwOnError:false});
      else node.textContent=equation;
      element.append(node);
    }
  }
  update(dimer,scale) {
    const changed=this.dimer?.id!==dimer.id;
    this.dimer=dimer;this.scale=scale;
    if(changed) {
      this.i=0;this.j=dimer.split;
      this.$("pairA").replaceChildren();this.$("pairB").replaceChildren();
      dimer.symbols.forEach((_,i)=>this.$(i<dimer.split?"pairA":"pairB").add(new Option(this.name(i),i)));
    }
    this.$("modelSummary").textContent=this.keys().map(key=>{
      const metric=this.data.metrics[key],m=this.data.models[key];
      return `${key}: ${m.channels===3?"C10/C20/C22c":m.arm==="shared-iso"?"f = 1":"C10/C20"} · MAE all ${this.f(metric.all.mae)} (n=${metric.all.n}), pair-disjoint ${this.f(metric.disjoint.mae)} (n=${metric.disjoint.n}) kcal/mol`;
    }).join("   |   ");
    this.refreshPair(changed);
    this.parameters();
  }
  refreshPair(reset) {
    const d=this.dimer,rec=d.records[this.scale],xyz=d.geometries[this.scale],v=this.viewer;
    this.$("pairA").value=this.i;this.$("pairB").value=this.j;
    const camera=v.getView();
    v.removeAllModels();v.removeAllShapes();v.removeAllLabels();
    const atoms=d.symbols.map((elem,i)=>({elem,x:xyz[i][0],y:xyz[i][1],z:xyz[i][2],index:i,bonds:[],bondOrder:[]}));
    for(const [a,b] of d.bonds){atoms[a].bonds.push(b);atoms[b].bonds.push(a);atoms[a].bondOrder.push(1);atoms[b].bondOrder.push(1);}
    const model=v.addModel();model.addAtoms(atoms);
    model.setStyle({},{stick:{radius:.12},sphere:{radius:.24}});
    model.setStyle({index:this.i},{stick:{radius:.12},sphere:{radius:.43,color:"#2563eb"}});
    model.setStyle({index:this.j},{stick:{radius:.12},sphere:{radius:.43,color:"#dc2626"}});
    model.setClickable({},true,atom=>{
      if(atom.index<d.split)this.i=atom.index;else this.j=atom.index;
      this.refreshPair(false);
    });
    const point=i=>({x:xyz[i][0],y:xyz[i][1],z:xyz[i][2]});
    v.addLine({start:point(this.i),end:point(this.j),color:"#b78d19",dashed:true,linewidth:2});
    for(const i of [this.i,this.j])v.addLabel(this.name(i),{position:point(i),fontColor:i<d.split?"#2563eb":"#dc2626",fontSize:12,showBackground:false});
    if(reset){v.zoomTo();v.zoom(1.5);}else v.setView(camera);
    v.render();
    const r=Math.hypot(...xyz[this.i].map((x,k)=>xyz[this.j][k]-x));
    this.$("pairCaption").textContent=`${this.name(this.i)} ↔ ${this.name(this.j)} · r = ${this.f(r)} Å · ${this.scale} Rₑ`;
    this.charts();this.equations();this.diagnostics();
  }
  breakdown(key,scale=this.scale) {
    return ModelMath.pair(this.dimer,this.dimer.records[scale],this.dimer.geometries[scale],
      this.data.models[key],this.i,this.j);
  }
  plot(target,title,xlabel,series) {
    const W=580,H=330,left=65,right=16,top=35,bottom=55;
    const points=series.flatMap(s=>s.points);
    let xmin=Math.min(...points.map(p=>p.x)),xmax=Math.max(...points.map(p=>p.x));
    let ymin=Math.min(0,...points.map(p=>p.y)),ymax=Math.max(...points.map(p=>p.y));
    if(xmax===xmin)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;
    const pad=(ymax-ymin)*.08;ymax+=pad;
    const X=x=>left+(x-xmin)/(xmax-xmin)*(W-left-right);
    const Y=y=>H-bottom-(y-ymin)/(ymax-ymin)*(H-top-bottom);
    const esc=s=>this.escape(s);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>
      <text x="${left}" y="17" fill="#eee" font-size="14">${esc(title)}</text>`;
    for(let i=0;i<=4;i++){
      const y=ymin+(ymax-ymin)*i/4,x=xmin+(xmax-xmin)*i/4;
      svg+=`<line x1="${left}" x2="${W-right}" y1="${Y(y)}" y2="${Y(y)}" stroke="#403b46"/>
      <text x="${left-7}" y="${Y(y)+4}" text-anchor="end" fill="#bbb" font-size="10">${this.f(y)}</text>
      <text x="${X(x)}" y="${H-bottom+19}" text-anchor="middle" fill="#bbb" font-size="10">${this.f(x)}</text>`;
    }
    for(const s of series){
      const sorted=[...s.points].sort((a,b)=>a.x-b.x);
      svg+=`<polyline fill="none" stroke="${s.color}" stroke-width="2" ${s.dashed?'stroke-dasharray="5 4"':""} points="${sorted.map(p=>`${X(p.x)},${Y(p.y)}`).join(" ")}"/>`;
      for(const p of sorted) svg+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="${p.scale===this.scale?5.5:3.5}" fill="${s.color}" stroke="${p.scale===this.scale?"#fff":s.color}" data-scale="${p.scale}" tabindex="0" role="button" aria-label="${esc(s.name)} at ${p.scale} R equilibrium" style="cursor:pointer"><title>${esc(s.name)} · ${p.scale} Rₑ · ${p.x.toPrecision(7)}, ${p.y.toPrecision(9)} kcal/mol</title></circle>`;
    }
    svg+=`<text x="${(W+left-right)/2}" y="${H-12}" text-anchor="middle" fill="#ccc" font-size="11">${esc(xlabel)}</text>
      <text transform="translate(13 ${(H+top-bottom)/2}) rotate(-90)" text-anchor="middle" fill="#ccc" font-size="11">Exchange / kcal mol⁻¹</text></svg>`;
    target.innerHTML=svg+`<p class="subtle">${series.map(s=>`<span style="color:${s.color}">${esc(s.name)}${s.dashed?" (dashed)":""}</span>`).join(" · ")}<br>Click a point to select that geometry; lines connect sampled points, not a fitted interpolation.</p>`;
    target.querySelectorAll("[data-scale]").forEach(node=>{
      node.onclick=()=>this.onScale(node.dataset.scale);
      node.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();this.onScale(node.dataset.scale);}};
    });
  }
  charts() {
    const scales=Object.keys(this.dimer.records).sort((a,b)=>Number(a)-Number(b));
    const colors=["#5fcab4","#c18ce3"],pairSeries=[],totalSeries=[];
    this.keys().forEach((key,i)=>{
      const points=scales.map(scale=>{
        const p=this.breakdown(key,scale);
        return {scale,x:p.r,y:this.dimer.records[scale].models[key].pairs[p.index],radial:p.radial};
      });
      pairSeries.push({name:key,color:colors[i],points});
      pairSeries.push({name:`${key}: fᵢ=fⱼ=1`,color:colors[i],dashed:true,points:points.map(p=>({...p,y:p.radial}))});
      totalSeries.push({name:key,color:colors[i],points:scales.map(scale=>({scale,x:Number(scale),y:this.dimer.records[scale].models[key].total}))});
    });
    totalSeries.push({name:"SAPT0/adz reference",color:"#eeeeee",points:scales.map(scale=>({scale,x:Number(scale),y:this.dimer.records[scale].target}))});
    for(const [key,color] of [["cliff_exch_5k","#e8b86a"],["cliff_exch_2k","#81818c"]])
      if(this.dimer.records[scales[0]].cliff[key]!==undefined)
        totalSeries.push({name:key==="cliff_exch_2k"?"CLIFF2 2k (collapsed)":"CLIFF2 5k",color,points:scales.map(scale=>({scale,x:Number(scale),y:this.dimer.records[scale].cliff[key]}))});
    this.plot(this.$("pairPlot"),`${this.name(this.i)} ↔ ${this.name(this.j)}`,"Actual atom-pair distance / Å",pairSeries);
    this.plot(this.$("totalPlot"),"Dimer total exchange","S66 separation / Rₑ",totalSeries);
  }
  equations() {
    const root=this.$("pairEquations");root.replaceChildren();
    for(const key of this.keys()){
      const p=this.breakdown(key),f=n=>this.f(n),model=this.data.models[key];
      const energy=this.dimer.records[this.scale].models[key].pairs[p.index];
      const card=document.createElement("section");card.className="equation-card";
      const title=document.createElement("h4");title.textContent=`${key} · ${this.name(this.i)} ↔ ${this.name(this.j)} · step ${model.step}`;
      title.className="font-semibold";card.append(title);
      const math=document.createElement("div");
      const sum=(params,h)=>model.arm==="shared-iso"?"1.00":
        "1.00"+h.slice(0,model.channels).map((v,k)=>`+(${f(params[k+2])})(${f(v)})`).join("");
      this.math(math,[
        `f_i=${sum(p.pa,p.hi)}=${f(p.fa)},\\qquad f_j=${sum(p.pb,p.hj)}=${f(p.fb)}`,
        `x=\\sqrt{(${f(p.pa[1])})(${f(p.pb[1])})}\\,(${f(p.r)})=${f(p.x)}`,
        `E_{ij}=(${f(p.pa[0])})(${f(p.pb[0])})(${f(p.fa)})(${f(p.fb)})\\left[1+${f(p.x)}+\\frac{${f(p.x)}^2}{3}\\right]e^{-${f(p.x)}}`,
        `E_{ij}=${f(energy)}\\;\\mathrm{kcal/mol},\\qquad E_{ij}^{f_i=f_j=1}=${f(p.radial)}\\;\\mathrm{kcal/mol}`
      ]);
      card.append(math);
      const table=document.createElement("div");table.className="table-wrap";
      table.innerHTML=`<table class="table table-xs"><thead><tr><th>Term</th><th>i coefficient</th><th>C̄ᵢ (+r̂)</th><th>i contribution</th><th>j coefficient</th><th>C̄ⱼ (−r̂)</th><th>j contribution</th></tr></thead><tbody>${
        ["10","20","22c"].slice(0,model.channels).map((name,k)=>`<tr><td>${name}</td><td>${f(p.pa[k+2])}</td><td>${f(p.hi[k])}</td><td>${f(p.pa[k+2]*p.hi[k])}</td><td>${f(p.pb[k+2])}</td><td>${f(p.hj[k])}</td><td>${f(p.pb[k+2]*p.hj[k])}</td></tr>`).join("")
      }</tbody></table>`;
      card.append(table);
      const note=document.createElement("p");note.className="subtle";
      note.textContent=`Full-precision Eᵢⱼ: ${energy.toPrecision(10)} kcal/mol. Numeric substitutions above are rounded, so re-evaluating them may not reproduce the full-precision result. Pair share: ${f(100*energy/this.dimer.records[this.scale].models[key].total)}% of dimer exchange.`;
      card.append(note);root.append(card);
    }
  }
  diagnostics() {
    const rec=this.dimer.records[this.scale],root=this.$("recordDiagnostics");
    root.innerHTML=`<p class="model-note">${this.escape(rec.system_id)} · ${rec.sharedPair?"training pair shared":"pair-disjoint"} · dropped transverse references: ${rec.degenerate}</p>`;
    for(const key of this.keys()){
      const m=rec.models[key],section=document.createElement("div");section.className="equation-card";
      section.textContent=`${key} · prediction ${this.f(m.total)} · reference ${this.f(rec.target)} · signed error ${this.f(m.total-rec.target)} kcal/mol · min f on intermonomer directions ${this.f(m.minAngular)} · negative pairs ${m.negativePairs}`;
      root.append(section);
    }
  }
  parameters() {
    const ids=this.$("allTypes").checked?Object.keys(this.data.types):
      [...new Set(this.dimer.records[this.scale].types)];
    const esc=s=>this.escape(s),f=n=>this.f(n);
    let html="<table class='table table-xs'><thead><tr><th>Model / type</th><th>Z</th><th>Effective mode</th><th>A</th><th>B / Å⁻¹</th><th>a10</th><th>a20</th><th>a22c</th><th>l=1 / 0.40</th><th>l=2 / 0.40</th></tr></thead><tbody>";
    for(const key of this.keys())for(const id of ids){
      const m=this.data.models[key],t=this.data.types[id],p=m.parameters[id],l1=Math.abs(p[2]),l2=Math.hypot(p[3],p[4]);
      html+=`<tr title="${esc(JSON.stringify(t.label))}"><td>${esc(key)} / ${id}</td><td>${t.z}</td><td>${esc(ModelMath.effectiveMode(m,t))}</td>${p.map(v=>`<td title="${v.toPrecision(12)}">${f(v)}</td>`).join("")}<td>${f(l1)} <progress class="progress norm" max=".4" value="${l1}"></progress></td><td>${f(l2)} <progress class="progress norm" max=".4" value="${l2}"></progress></td></tr>`;
    }
    this.$("parameterTable").innerHTML=html+"</tbody></table><p class='subtle'>Hover a row for its full chemical type label; hover a parameter for more digits. Values are bounded derived parameters, not raw checkpoint tensors. A: √(kcal/mol).</p>";
  }
}
