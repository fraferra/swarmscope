"""Local read-only dashboard: stdlib ``http.server`` + one HTML page.

Lineage tree, cost sunburst, dedup heatmap over time, value-vs-k curve with
CIs, and — on every view — the unknown-lineage fraction and whether the
waste ratio is even defined.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..attribution.graph import LineageGraph
from ..attribution.rollup import cost_rollup
from ..attribution.waste import waste_report
from ..evaluation.proxies import online_proxies
from ..store.base import Store, open_store

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>swarmscope</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#0f1117;color:#e6e6e6}
header{padding:10px 16px;background:#161a24;display:flex;gap:16px;align-items:center}
header select{background:#222;color:#eee;border:1px solid #444;padding:4px}
main{padding:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.card{background:#161a24;border:1px solid #262b38;border-radius:8px;padding:12px;min-height:120px;min-width:0;overflow-x:auto}
svg{max-width:100%;height:auto}
.card h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#9aa4b2}
.kpi{display:flex;gap:18px;flex-wrap:wrap}.kpi div{min-width:110px}.kpi b{display:block;font-size:22px}
.warn{color:#ffb454}.bad{color:#ff6b6b}.ok{color:#7ee787}
ul.tree{list-style:none;padding-left:14px;margin:0}ul.tree li{margin:2px 0;cursor:pointer}
ul.tree .cost{color:#9aa4b2;font-size:12px}
svg text{fill:#ccc;font-size:11px}
.full{grid-column:1/-1}
small{color:#9aa4b2}
</style></head><body>
<header><b>swarmscope</b> <select id="run"></select> <span id="meta"></span> <a href="/reputation" style="margin-left:auto;color:#7ee787">bandit →</a></header>
<main>
<div class="card full"><h3>Run</h3><div class="kpi" id="kpi"></div><div id="warnings"></div></div>
<div class="card"><h3>Cost sunburst (subtree cost, by lineage)</h3><svg id="sun" width="460" height="460" viewBox="0 0 460 460"></svg></div>
<div class="card"><h3>Lineage tree</h3><div id="tree" style="max-height:440px;overflow:auto"></div></div>
<div class="card"><h3>Dedup hit rate over time</h3><svg id="heat" width="460" height="140" viewBox="0 0 460 140"></svg><small id="heatnote"></small></div>
<div class="card"><h3>Value vs k (P(success), 95% CI)</h3><svg id="curve" width="460" height="260" viewBox="0 0 460 260"></svg><small id="curvenote"></small></div>
<div class="card full"><h3>Per-source verdicts &amp; waste</h3><div id="sources"></div></div>
<div class="card full"><h3>Route reputation (store-wide; P(success) with 95% CI)</h3><div id="rep"></div></div>
</main>
<script>
const $=s=>document.querySelector(s);const fmt=(x,d=2)=>x==null?'—':(typeof x==='number'?x.toFixed(d):x);
async function j(u){const r=await fetch(u);return r.json()}
async function loadRuns(){const runs=await j('/api/runs');const sel=$('#run');sel.innerHTML='';
 for(const r of runs){const o=document.createElement('option');o.value=r.run_id;o.textContent=(r.name||r.run_id)+' ('+r.event_count+' ev)';sel.appendChild(o)}
 sel.onchange=()=>show(sel.value);if(runs.length)show(runs[0].run_id)}
async function show(id){const [cost,waste,prox,curve]=await Promise.all([j('/api/run/'+id+'/cost'),j('/api/run/'+id+'/waste'),j('/api/run/'+id+'/proxies'),j('/api/run/'+id+'/ablation')]);
 const t=cost.totals;const ul=cost.unknown_lineage_fraction;
 $('#kpi').innerHTML=`<div><small>agents</small><b>${t.agents}</b></div><div><small>tokens</small><b>${t.tokens}</b></div>
 <div><small>cost USD</small><b>$${fmt(t.cost_usd,4)}</b></div><div><small>unknown lineage</small><b class="${ul>0.2?'bad':ul>0?'warn':'ok'}">${(ul*100).toFixed(1)}%</b></div>
 <div><small>waste ratio</small><b class="${waste.defined?'':'warn'}">${waste.defined?(waste.waste_ratio*100).toFixed(1)+'%':'undefined'}</b></div>
 <div><small>accepted artifacts</small><b>${waste.accepted_artifacts}</b></div><div><small>novel claims / agent-h</small><b>${fmt(prox.novel_claim_rate_per_agent_hour,1)}</b></div>
 <div><small>dedup hit rate</small><b>${prox.dedup_hit_rate==null?'—':(prox.dedup_hit_rate*100).toFixed(0)+'%'}</b></div>
 <div><small>coverage entropy</small><b>${fmt(prox.coverage_entropy)}</b></div><div><small>unpriced gens</small><b class="${cost.unpriced_generations?'warn':''}">${cost.unpriced_generations}</b></div>`;
 const w=[...(prox.warnings||[])];if(!waste.defined)w.push(waste.reason);$('#warnings').innerHTML=w.map(x=>`<div class="warn">⚠ ${x}</div>`).join('');
 tree(cost.tree);sunburst(cost.tree,t.cost_usd);heat(prox.dedup_hit_rate_over_time);curveChart(curve);sources(waste);reputation()}
async function reputation(){const r=await j('/api/reputation');const el=$('#rep');if(!r.leaderboard||!r.leaderboard.length){el.innerHTML='<small>no reputation yet — wrap work in <code>with sdk.request(...)</code> and emit verdicts</small>';return}
 el.innerHTML=`<table style="border-collapse:collapse"><tr><th align=left>route</th><th>P(success)</th><th>95% CI</th><th>evidence</th></tr>${r.leaderboard.map(x=>`<tr><td>${x.route.join(' → ')}</td><td align=center>${x.mean.toFixed(2)}</td><td align=center>[${x.ci_low.toFixed(2)}, ${x.ci_high.toFixed(2)}]</td><td align=center>${(x.global_successes+x.global_failures).toFixed(1)}</td></tr>`).join('')}</table>`}
function tree(nodes){const el=$('#tree');el.innerHTML='';const build=(ns)=>{const ul=document.createElement('ul');ul.className='tree';
 for(const n of ns){const li=document.createElement('li');li.innerHTML=`${n.children.length?'▸ ':'• '}${n.name} <span class="cost">$${fmt(n.subtree_cost_usd,4)} · ${n.subtree_tokens} tok${n.group_id?' · '+n.group_id:''}</span>`;
 if(n.children.length){const c=build(n.children);c.style.display='none';li.appendChild(c);li.onclick=e=>{e.stopPropagation();c.style.display=c.style.display==='none'?'':'none'}}ul.appendChild(li)}return ul};el.appendChild(build(nodes))}
function sunburst(nodes,total){const svg=$('#sun');svg.innerHTML='';const cx=230,cy=230,R=215,depthMax=6;if(!total){svg.innerHTML='<text x="20" y="30">no priced cost</text>';return}
 const arc=(r0,r1,a0,a1)=>{const p=(r,a)=>[cx+r*Math.cos(a),cy+r*Math.sin(a)];const [x0,y0]=p(r1,a0),[x1,y1]=p(r1,a1),[x2,y2]=p(r0,a1),[x3,y3]=p(r0,a0);const L=a1-a0>Math.PI?1:0;
 return `M${x0},${y0}A${r1},${r1},0,${L},1,${x1},${y1}L${x2},${y2}A${r0},${r0},0,${L},0,${x3},${y3}Z`};
 const col=i=>`hsl(${(i*47)%360},60%,${45}%)`;let i=0;
 const rec=(ns,a0,a1,d)=>{if(d>depthMax)return;let a=a0;const sum=ns.reduce((s,n)=>s+Math.max(n.subtree_cost_usd,0),0)||1;for(const n of ns){const span=(a1-a0)*Math.max(n.subtree_cost_usd,0)/sum;if(span<0.002){a+=span;continue}
 const r0=(R/(depthMax+1))*d,r1=(R/(depthMax+1))*(d+1)-1;const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d',arc(r0,r1,a,a+span));path.setAttribute('fill',col(i++));path.setAttribute('stroke','#0f1117');
 const t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=`${n.name}: $${fmt(n.subtree_cost_usd,4)} (${(100*n.subtree_cost_usd/total).toFixed(1)}%)`;path.appendChild(t);svg.appendChild(path);rec(n.children,a,a+span,d+1);a+=span}};
 rec(nodes,-Math.PI/2,1.5*Math.PI,0)}
function heat(rows){const svg=$('#heat');svg.innerHTML='';if(!rows||!rows.length){svg.innerHTML='<text x="10" y="30">no claim lookups recorded</text>';$('#heatnote').textContent='';return}
 const w=440/rows.length;rows.forEach((r,i)=>{const v=r.hit_rate==null?0:r.hit_rate;const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');rect.setAttribute('x',10+i*w);rect.setAttribute('y',20);rect.setAttribute('width',Math.max(1,w-1));rect.setAttribute('height',80);
 rect.setAttribute('fill',`hsl(${120-120*v},70%,${25+35*v}%)`);const t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=`t=${r.t_start_s}s: ${r.hits}/${r.claims} hits`;rect.appendChild(t);svg.appendChild(rect)});
 $('#heatnote').textContent=`${rows.length} windows; green=novel, red=duplicate`}
function curveChart(c){const svg=$('#curve');svg.innerHTML='';const note=$('#curvenote');if(!c||!c.points||!c.points.length){svg.innerHTML='<text x="10" y="30">no ablation stored — run `swarmscope ablate`</text>';note.textContent=c&&c.notes?c.notes.join(' '):'';return}
 const W=460,H=260,ml=40,mb=30;const ks=c.points.map(p=>p.k);const lx=k=>ml+(Math.log(k)/Math.log(Math.max(...ks)||2))*(W-ml-10);const ly=p=>H-mb-p*(H-mb-15);
 const ns='http://www.w3.org/2000/svg';const band=document.createElementNS(ns,'path');let d='M'+c.points.map(p=>lx(p.k)+','+ly(p.ci_high)).join('L')+'L'+[...c.points].reverse().map(p=>lx(p.k)+','+ly(p.ci_low)).join('L')+'Z';band.setAttribute('d',d);band.setAttribute('fill','rgba(126,231,135,.2)');svg.appendChild(band);
 const line=document.createElementNS(ns,'path');line.setAttribute('d','M'+c.points.map(p=>lx(p.k)+','+ly(p.p_success)).join('L'));line.setAttribute('stroke','#7ee787');line.setAttribute('fill','none');line.setAttribute('stroke-width','2');svg.appendChild(line);
 for(const p of c.points){const t=document.createElementNS(ns,'text');t.setAttribute('x',lx(p.k)-6);t.setAttribute('y',H-8);t.textContent=p.k;svg.appendChild(t)}
 for(const v of [0,0.5,1]){const t=document.createElementNS(ns,'text');t.setAttribute('x',4);t.setAttribute('y',ly(v)+4);t.textContent=v;svg.appendChild(t)}
 if(c.knee_k){const kx=lx(c.knee_k);const l=document.createElementNS(ns,'line');l.setAttribute('x1',kx);l.setAttribute('x2',kx);l.setAttribute('y1',10);l.setAttribute('y2',H-mb);l.setAttribute('stroke','#ffb454');l.setAttribute('stroke-dasharray','4 3');svg.appendChild(l)}
 note.textContent=`unit=${c.unit}, n=${c.n_units}, knee≈${c.knee_k??'—'}${c.fidelity&&!c.fidelity.identical?' · replay differs from original (approximate)':''} ${(c.notes||[]).join(' ')}`}
function sources(w){const rows=Object.entries(w.by_source||{});const vs=w.verdict_sources||{};
 $('#sources').innerHTML=`<div><small>verdicts by provenance: ${Object.entries(vs).map(([k,v])=>k+'='+v).join(', ')||'none'}</small></div>`+
 (rows.length?`<table style="border-collapse:collapse;margin-top:6px">${'<tr><th align=left>source</th><th>accepted</th><th>contributing agents</th><th>waste ratio</th></tr>'}${rows.map(([s,r])=>`<tr><td>${s}</td><td align=center>${r.accepted_artifacts}</td><td align=center>${r.contributing_agents}</td><td align=center>${r.waste_ratio==null?'—':(r.waste_ratio*100).toFixed(1)+'%'}</td></tr>`).join('')}</table>`:'')+
 (w.cost_per_accepted_dist&&w.cost_per_accepted_dist.p50!=null?`<div style="margin-top:6px"><small>cost per accepted artifact: p50=$${fmt(w.cost_per_accepted_dist.p50,4)} p90=$${fmt(w.cost_per_accepted_dist.p90,4)} max=$${fmt(w.cost_per_accepted_dist.p100,4)} (distribution, not a mean)</small></div>`:'')}
loadRuns();
</script></body></html>"""


REPUTATION_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>swarmscope · bandit</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#0f1117;color:#e6e6e6}
header{padding:10px 16px;background:#161a24;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header a{color:#7ee787}header select,header input{background:#222;color:#eee;border:1px solid #444;padding:4px}
main{padding:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.card{background:#161a24;border:1px solid #262b38;border-radius:8px;padding:12px;min-width:0;overflow-x:auto}
svg{max-width:100%;height:auto}
.card h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#9aa4b2}
.full{grid-column:1/-1}table{border-collapse:collapse;width:100%}th,td{padding:4px 8px;border-bottom:1px solid #262b38;font-size:13px}
th{text-align:left;color:#9aa4b2}td.num{text-align:right;font-variant-numeric:tabular-nums}
svg text{fill:#ccc;font-size:11px}small{color:#9aa4b2}.bar{height:8px;background:#2a2f3d;border-radius:4px;position:relative}
.bar i{position:absolute;top:0;bottom:0;background:rgba(126,231,135,.35)}.bar b{position:absolute;top:-2px;width:2px;height:12px;background:#7ee787}
input[type=text]{width:60%}button{background:#2a2f3d;color:#eee;border:1px solid #444;padding:4px 10px;cursor:pointer}
</style></head><body>
<header><b>swarmscope · bandit</b> <a href="/">← runs</a>
 <label>unit <select id="unit"><option value="sequence">sequence</option><option value="agent">agent</option></select></label>
 <span id="summary"></span></header>
<main>
<div class="card full"><h3>Learnt success probabilities (Beta posterior per route; bar = 95% credible interval, tick = mean)</h3><div id="table"></div></div>
<div class="card"><h3>Posterior densities (top 8 routes)</h3><svg id="dens" width="460" height="260" viewBox="0 0 460 260"></svg><small id="densnote"></small></div>
<div class="card"><h3>Evidence over time (cumulative successes − failures per route)</h3><svg id="time" width="460" height="260" viewBox="0 0 460 260"></svg><small id="timenote"></small></div>
<div class="card full"><h3>Ask the bandit</h3>
 <input type="text" id="q" placeholder="a request, e.g. translate this clause to French"> kind <input type="text" id="kind" size="8" value="request">
 policy <select id="policy"><option>thompson</option><option>ucb</option><option>greedy</option></select> <button id="ask">advise</button>
 <div id="advice" style="margin-top:8px"></div></div>
</main>
<script>
const $=s=>document.querySelector(s);const f2=x=>x==null?'—':x.toFixed(2);
async function j(u){const r=await fetch(u);return r.json()}
function lgamma(x){const c=[76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5];let y=x,t=x+5.5;t-=(x+0.5)*Math.log(t);let s=1.000000000190015;for(let i=0;i<6;i++)s+=c[i]/++y;return -t+Math.log(2.5066282746310005*s/x)}
function betapdf(x,a,b){if(x<=0||x>=1)return 0;return Math.exp((a-1)*Math.log(x)+(b-1)*Math.log(1-x)+lgamma(a+b)-lgamma(a)-lgamma(b))}
const col=i=>`hsl(${(i*57)%360},65%,60%)`;
async function load(){const unit=$('#unit').value;const [rep,tl]=await Promise.all([j('/api/reputation?unit='+unit),j('/api/reputation/outcomes')]);
 const rows=rep.leaderboard;$('#summary').innerHTML=`<small>${rows.length} routes · ${tl.total} outcomes · ${tl.requests} requests</small>`;
 if(!rows.length){$('#table').innerHTML='<small>no reputation yet — wrap work in <code>with sdk.request(...)</code>, emit verdicts, or run <code>swarmscope reputation --rebuild</code></small>';$('#dens').innerHTML='';$('#time').innerHTML='';return}
 $('#table').innerHTML=`<table><tr><th>route</th><th>P(success)</th><th>95% CI</th><th style="width:30%">posterior</th><th>successes</th><th>failures</th></tr>${rows.map((r,i)=>`<tr><td><span style="color:${col(i)}">●</span> ${r.route.join(' → ')}</td><td class=num>${f2(r.mean)}</td><td class=num>[${f2(r.ci_low)}, ${f2(r.ci_high)}]</td><td><div class=bar><i style="left:${r.ci_low*100}%;width:${(r.ci_high-r.ci_low)*100}%"></i><b style="left:${r.mean*100}%"></b></div></td><td class=num>${r.global_successes.toFixed(1)}</td><td class=num>${r.global_failures.toFixed(1)}</td></tr>`).join('')}</table>`;
 dens(rows.slice(0,8));timeline(tl,unit)}
function dens(rows){const svg=$('#dens');svg.innerHTML='';const W=460,H=260,ml=30,mb=24;const ns='http://www.w3.org/2000/svg';let ymax=0;const curves=rows.map(r=>{const a=1+r.global_successes,b=1+r.global_failures;const pts=[];for(let i=1;i<200;i++){const x=i/200;const y=betapdf(x,a,b);ymax=Math.max(ymax,y);pts.push([x,y])}return pts});
 curves.forEach((pts,i)=>{const p=document.createElementNS(ns,'path');p.setAttribute('d','M'+pts.map(([x,y])=>(ml+x*(W-ml-10))+','+(H-mb-y/ymax*(H-mb-15))).join('L'));p.setAttribute('stroke',col(i));p.setAttribute('fill','none');p.setAttribute('stroke-width','1.8');const t=document.createElementNS(ns,'title');t.textContent=rows[i].route.join(' → ');p.appendChild(t);svg.appendChild(p)});
 for(const v of [0,0.25,0.5,0.75,1]){const t=document.createElementNS(ns,'text');t.setAttribute('x',ml+v*(W-ml-10)-6);t.setAttribute('y',H-8);t.textContent=v;svg.appendChild(t)}
 $('#densnote').textContent='Beta(1+successes, 1+failures) — a tall narrow peak is a well-measured route; a wide hump has been tried a few times'}
function timeline(tl,unit){const svg=$('#time');svg.innerHTML='';const ns='http://www.w3.org/2000/svg';const W=460,H=260,ml=30,mb=24;
 const byRoute={};for(const o of tl.outcomes){const key=unit==='agent'?o.route[o.route.length-1]:o.route.join(' → ');(byRoute[key]=byRoute[key]||[]).push(o)}
 const keys=Object.keys(byRoute).slice(0,8);if(!keys.length)return;const t0=Math.min(...tl.outcomes.map(o=>o.ts)),t1=Math.max(...tl.outcomes.map(o=>o.ts))||t0+1;let lo=0,hi=0;const series=keys.map(k=>{let c=0;const pts=byRoute[k].sort((a,b)=>a.ts-b.ts).map(o=>{c+=o.accepted?o.weight:-o.weight;lo=Math.min(lo,c);hi=Math.max(hi,c);return [o.ts,c]});return pts});
 const sx=t=>ml+((t-t0)/((t1-t0)||1))*(W-ml-10),sy=v=>H-mb-((v-lo)/((hi-lo)||1))*(H-mb-15);
 series.forEach((pts,i)=>{const p=document.createElementNS(ns,'path');p.setAttribute('d','M'+sx(t0)+','+sy(0)+'L'+pts.map(([t,v])=>sx(t)+','+sy(v)).join('L'));p.setAttribute('stroke',col(i));p.setAttribute('fill','none');p.setAttribute('stroke-width','1.8');const tt=document.createElementNS(ns,'title');tt.textContent=keys[i];p.appendChild(tt);svg.appendChild(p)});
 const z=document.createElementNS(ns,'line');z.setAttribute('x1',ml);z.setAttribute('x2',W-10);z.setAttribute('y1',sy(0));z.setAttribute('y2',sy(0));z.setAttribute('stroke','#444');svg.appendChild(z);
 $('#timenote').textContent=`${tl.outcomes.length} most recent outcomes; weight = verdict-source weight × confidence`}
$('#ask').onclick=async()=>{const q=$('#q').value.trim();if(!q)return;const a=await j('/api/reputation/advise?q='+encodeURIComponent(q)+'&kind='+encodeURIComponent($('#kind').value)+'&policy='+$('#policy').value+'&unit='+$('#unit').value);
 if(a.cold_start){$('#advice').innerHTML=`<small>cold start${a.timed_out?' (timed out)':''}: no evidence and no registered candidates — the caller would use its default route</small>`;return}
 $('#advice').innerHTML=`<small>${a.similar.length} similar past requests · policy ${a.policy}</small><table><tr><th>route</th><th>score</th><th>P(success)</th><th>95% CI</th><th>local evidence</th><th>global evidence</th><th>similar</th></tr>${a.routes.map(r=>`<tr><td>${r.route.join(' → ')}</td><td class=num>${f2(r.score)}</td><td class=num>${f2(r.mean)}</td><td class=num>[${f2(r.ci_low)}, ${f2(r.ci_high)}]</td><td class=num>${r.local_successes.toFixed(1)} / ${r.n.toFixed(1)}</td><td class=num>${r.global_successes.toFixed(1)} / ${(r.global_successes+r.global_failures).toFixed(1)}</td><td class=num>${r.similar_requests}</td></tr>`).join('')}</table>`};
$('#unit').onchange=load;load();
</script></body></html>"""


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            try:
                if not parts:
                    body = HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parts[:2] == ["api", "runs"]:
                    return self._json([r.to_dict() for r in store.runs()])
                if parts == ["reputation"]:
                    body = REPUTATION_HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parts[:2] == ["api", "reputation"]:
                    from ..reputation import Router, RouterPolicy

                    q = parse_qs(u.query)
                    unit = q.get("unit", ["sequence"])[0]
                    if len(parts) == 2:
                        lb = Router(store).leaderboard(limit=50, unit=unit)
                        return self._json({"leaderboard": [r.to_dict() for r in lb]})
                    if parts[2] == "outcomes":
                        outs = store.route_outcomes(limit=2000)
                        return self._json({"total": len(outs), "requests": len({o.request_id for o in outs}),
                                           "outcomes": [{"ts": o.ts, "route": o.route, "accepted": o.accepted,
                                                         "weight": o.weight, "source": o.source, "request_id": o.request_id}
                                                        for o in outs]})
                    if parts[2] == "advise":
                        text = q.get("q", [""])[0]
                        pol = RouterPolicy(policy=q.get("policy", ["greedy"])[0], unit=unit)
                        r = Router(store, policy=pol)
                        for st in store.route_stats():  # every known route is a candidate
                            r.register_identity(st.route[-1] if unit == "agent" else " > ".join(st.route))
                        adv = r.advise(text, request_id="_dash", kind=q.get("kind", ["request"])[0])
                        return self._json(adv.to_dict())
                if parts[:2] == ["api", "run"] and len(parts) == 4:
                    run_id, what = parts[2], parts[3]
                    g = LineageGraph(store.events(run_id))
                    if what == "cost":
                        return self._json(cost_rollup(store, run_id, g).to_dict())
                    if what == "waste":
                        q = parse_qs(u.query)
                        srcs = q.get("sources", ["human,verifier,judge"])[0].split(",")
                        return self._json(waste_report(store, run_id, sources=srcs, graph=g).to_dict())
                    if what == "proxies":
                        return self._json(online_proxies(store, run_id, graph=g).to_dict())
                    if what == "ablation":
                        return self._json(store.calibration_get(f"ablation:{run_id}") or {})
                    if what == "events":
                        return self._json([e.to_dict() for e in store.events(run_id, limit=5000)])
                self._json({"error": "not found"}, 404)
            except Exception as exc:  # pragma: no cover
                self._json({"error": repr(exc)}, 500)

    return Handler


def serve(store: str | Store | None, host: str = "127.0.0.1", port: int = 8765, *, block: bool = True):
    st = open_store(store)
    httpd = ThreadingHTTPServer((host, port), make_handler(st))
    if block:
        print(f"swarmscope dashboard: http://{host}:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover
            pass
        finally:
            httpd.server_close()
        return None
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
