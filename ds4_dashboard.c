/* ds4_dashboard.c — see ds4_dashboard.h.
 *
 * A loopback-only HTTP server on a background thread. It renders nothing
 * itself: the page asks for /stats.json (built by ds4_profiler_json) and draws
 * it. Keeping the server and the renderer separate is what makes the view
 * extensible to N devices without touching C.
 */
#include "ds4_dashboard.h"
#include "ds4_profiler.h"

#ifdef _WIN32
/* No loopback server on the Windows build; the dashboard is a no-op there. */
int ds4_dashboard_active(void) { return 0; }
const char *ds4_dashboard_url(void) { return ""; }
int ds4_dashboard_start(const char *bind_addr, int port, int open_browser) {
    (void)bind_addr;
    (void)port; (void)open_browser;
    return 0;
}
void ds4_dashboard_stop(void) {}
#else

#include <arpa/inet.h>
#include <errno.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define DS4_DASH_DEFAULT_PORT 8765
#define DS4_DASH_PORT_TRIES   16

/* ------------------------------------------------------------------ page -- */

/* Single page, no external assets. Labels spell everything out (no acronyms)
 * and every number the engine exposes is shown in its tier. */
static const char DASHBOARD_HTML[] =
"<!doctype html>\n"
"<html lang='en'>\n"
"<head>\n"
"<meta charset='utf-8'>\n"
"<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
"<title>ds4 engine dashboard</title>\n"
"<style>\n"
":root{color-scheme:dark}\n"
"*{box-sizing:border-box}\n"
"body{margin:0;font:14px/1.45 system-ui,sans-serif;background:#0f1216;color:#e6edf3}\n"
"header{padding:16px 20px;border-bottom:1px solid #223;display:flex;align-items:baseline;gap:12px;position:sticky;top:0;background:#0f1216}\n"
"h1{font-size:18px;margin:0}\n"
"h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#8b98a5;margin:0 0 8px}\n"
".muted{color:#8b98a5;font-size:12px}\n"
"main{padding:16px 20px 48px;display:grid;gap:20px}\n"
".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}\n"
".card{background:#161b22;border:1px solid #223;border-radius:10px;padding:14px}\n"
".hero{font-size:34px;font-weight:650;margin:2px 0}\n"
".unit{font-size:14px;color:#8b98a5;font-weight:400}\n"
".split{display:flex;flex-wrap:wrap;gap:18px;margin-top:8px}\n"
".split div{font-size:12px;color:#8b98a5}\n"
".split b{display:block;font-size:15px;color:#e6edf3;font-weight:600}\n"
"table{width:100%;border-collapse:collapse}\n"
"th,td{text-align:right;padding:7px 8px;border-bottom:1px solid #1d2430}\n"
"th:first-child,td:first-child{text-align:left}\n"
"th{color:#8b98a5;font-weight:500;font-size:12px}\n"
".bar{height:8px;background:#223;border-radius:4px;overflow:hidden;margin:8px 0 2px}\n"
".bar>i{display:block;height:100%;background:#3b82f6}\n"
".stack{display:flex;height:14px;border-radius:4px;overflow:hidden;margin:8px 0 6px;background:#223}\n"
".stack>i{display:block;height:100%}\n"
"canvas{width:100%;display:block;margin-top:6px}\n"
".legend b{font-weight:600}\n"
".devhead{display:flex;justify-content:space-between;align-items:baseline;gap:10px}\n"
".tag{font-size:11px;padding:2px 8px;border-radius:999px;background:#1f2a3a;color:#9fc0ee;white-space:nowrap}\n"
".comp{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-top:10px}\n"
".comp div{background:#12171d;border:1px solid #1d2430;border-radius:8px;padding:8px}\n"
".comp span{display:block;font-size:11px;color:#8b98a5}\n"
".comp b{font-size:14px;font-weight:600}\n"
".legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:#8b98a5;margin-top:8px}\n"
"</style>\n"
"</head>\n"
"<body>\n"
"<header><h1>ds4 engine dashboard</h1><span class='muted' id='status'>connecting</span></header>\n"
"<main>\n"
"<section><h2>Throughput</h2><div class='cards' id='throughput'></div>\n"
"<div class='card' style='margin-top:12px'><div class='muted'>Speed over the last 180 samples (rolling min/max/avg; x: samples, newest right)</div>\n"
"<canvas id='tpsChart' height='150'></canvas><div class='legend' id='tpsLegend'></div></div></section>\n"
"<section><h2>Token cost</h2>\n"
"<div class='card'><div id='costBar'></div><table id='breakdown'></table>\n"
"<p class='muted' style='margin:10px 0 0'>Rows marked <b>total</b> add up; indented rows are the parts of the row above them. Rows marked <b>overlaps</b> run at the same time as the rest (data is moving while the processor works), so they are shown for reference and are not part of the total.</p></div>\n"
"<div class='card' style='margin-top:12px'><div class='muted'>Per-step timeline &mdash; one row per kind of cost, never two merged. Hover a row for what it means.\n"
"<label style='margin-left:10px;float:right'><input type='checkbox' id='advSteps'> show every step</label></div>\n"
"<table id='steps'></table></div>\n"
"<div class='card' style='margin-top:12px'><div class='muted'>Per-token trace &mdash; what every GPU was doing while the token was generated, which experts were used and where they came from, and the ordered steps.<label style='margin-left:10px;float:right'><input type='checkbox' id='advEvents'> all events</label></div>\n"
"<div id='tokList'></div><div id='tokEvents' style='margin-top:10px'></div></div>\n"
"</section>\n"
"<section><h2>Data movement (real time)</h2><div class='card' id='storage'></div>\n"
"<div class='card' style='margin-top:12px'><div class='muted'>PCI Express throughput per device (x: samples, newest right)</div>\n"
"<canvas id='pcieChart' height='150'></canvas><div class='legend' id='pcieLegend'></div></div></section>\n"
"<section><h2>Experts across the tiers</h2><div class='card' id='experts'></div></section>\n"
"<section><h2>Devices</h2><div class='cards' id='devices'></div></section>\n"
"</main>\n"
"<script>\n"
"const $=id=>document.getElementById(id);\n"
"function fmtB(v){v=Number(v)||0;const u=['B','KiB','MiB','GiB','TiB'];let i=0;while(Math.abs(v)>=1024&&i<u.length-1){v/=1024;i++;}return (i===0?v.toFixed(0):v.toFixed(2))+' '+u[i];}\n"
"const fmtMs=v=>(Number(v)||0).toFixed(1)+' ms';\n"
"const fmtTps=v=>(Number(v)||0).toFixed(1)+' tok/s';\n"
"const fmtTflops=v=>((Number(v)||0)/1e12).toFixed(1)+' TFLOP/s';\n"
"const fmtRate=v=>fmtB(v)+'/s';\n"
"const fmtGBps=v=>((Number(v)||0)/1e9).toFixed(2)+' GB/s';\n"
"const pctf=(a,b)=>b>0?100*a/b:0;\n"
"const pctstr=(a,b)=>(b>0?100*a/b:0).toFixed(0)+'%';\n"
"const stat=(t,k)=>(t&&t[k])||{last:0,min:0,max:0,average:0,samples:0};\n"
"const sub=(a,b)=>({last:Math.max(0,a.last-b.last),min:Math.max(0,a.min-b.min),max:Math.max(0,a.max-b.max),average:Math.max(0,a.average-b.average)});\n"
"function card(title,hero,parts){return `<div class='card'><div class='muted'>${title}</div><div class='hero'>${hero}</div>${parts?`<div class='split'>${parts}</div>`:''}</div>`;}\n"
"function statLine(s,f){return `<div>last<b>${f(s.last)}</b></div><div>min<b>${f(s.min)}</b></div><div>max<b>${f(s.max)}</b></div><div>avg<b>${f(s.average)}</b></div>`;}\n"
"function stack(el,items,title,fmt){fmt=fmt||fmtB;const tot=items.reduce((a,x)=>a+Math.max(0,x.value),0)||1;\n"
"  el.innerHTML=(title?`<div class='muted'>${title}</div>`:'')+\n"
"    `<div class='stack'>`+items.map(x=>`<i style='width:${(100*Math.max(0,x.value)/tot).toFixed(2)}%;background:${x.color}'></i>`).join('')+`</div>`+\n"
"    `<div class='legend'>`+items.map(x=>`<span><b style='color:${x.color}'>&#9632;</b> ${x.label} <b>${pctstr(x.value,tot)}</b> <span class='muted'>${fmt(x.value)}</span></span>`).join('')+`</div>`;}\n"
"function legend(el,items){el.innerHTML=items.map(x=>`<span><b style='color:${x.color}'>&#9632;</b> ${x.label}</span>`).join('');}\n"
"const HIST=180,hist={tps:[],tpsMin:[],tpsMax:[],tpsAvg:[],pre:[],pcie:{}};\n"
"function push(a,v){a.push(Number(v)||0);while(a.length>HIST)a.shift();}\n"
"function winmin(a,n){const s=a.slice(-n);return s.length?Math.min.apply(null,s):0;}\n"
"function winmax(a,n){const s=a.slice(-n);return s.length?Math.max.apply(null,s):0;}\n"
"function winavg(a,n){const s=a.slice(-n);return s.length?s.reduce((x,y)=>x+y,0)/s.length:0;}\n"
"function chart(cv,series,yfmt){const w=cv.clientWidth||600,h=150;cv.width=w;cv.height=h;\n"
"  const g=cv.getContext('2d');g.clearRect(0,0,w,h);\n"
"  let mx=0;for(const s of series)for(const v of s.data)if(v>mx)mx=v;\n"
"  if(mx<=0){g.fillStyle='#8b98a5';g.font='12px system-ui';g.fillText('collecting samples...',8,20);return;}\n"
"  g.strokeStyle='#1d2430';g.lineWidth=1;\n"
"  for(let i=0;i<=3;i++){const y=8+(h-24)*i/3;g.beginPath();g.moveTo(0,y);g.lineTo(w,y);g.stroke();}\n"
"  let n=2;for(const s of series)if(s.data.length>n)n=s.data.length;\n"
"  const X=i=>1+i*(w-2)/(n-1),Y=v=>h-12-(h-24)*(v/mx);\n"
"  for(const s of series){if(s.data.length<2)continue;g.strokeStyle=s.color;g.lineWidth=s.width||1.6;g.globalAlpha=s.alpha||1;g.beginPath();\n"
"    s.data.forEach((v,i)=>{i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v));});g.stroke();g.globalAlpha=1;}\n"
"  g.fillStyle='#8b98a5';g.font='11px system-ui';g.fillText('peak '+yfmt(mx),6,12);}\n"
"function role(t){return t===0?'Primary: dense weights, key/value cache, compute':'Expert tier: hot routed experts kept resident';}\n"
"let prevDev={},prevTime=0,prevHost=null,prevSsd=0,ssdPeak=0;\n"
"function render(d){\n"
"  const t=d.timing||{},h=d.host||{};\n"
"  $('status').textContent=(t.tokens||0)+' tokens measured';\n"
"  const tps=stat(t,'tokens_per_s'),ptt=stat(t,'per_token_ms'),pf=stat(t,'prefill_tokens_per_s');\n"
"  $('throughput').innerHTML=\n"
"    card('Decode speed',fmtTps(tps.average),statLine(tps,fmtTps))+\n"
"    card('Time per token',fmtMs(ptt.average),statLine(ptt,fmtMs))+\n"
"    card('Prompt processing',fmtTps(pf.average),statLine(pf,fmtTps));\n"
"  push(hist.tps,tps.last);push(hist.pre,pf.last);\n"
"  push(hist.tpsMin,winmin(hist.tps,20));push(hist.tpsMax,winmax(hist.tps,20));push(hist.tpsAvg,winavg(hist.tps,20));\n"
"  chart($('tpsChart'),[\n"
"    {color:'#1d4ed8',data:hist.tpsMax,width:1,alpha:.6,label:'decode max'},\n"
"    {color:'#1d4ed8',data:hist.tpsMin,width:1,alpha:.6,label:'decode min'},\n"
"    {color:'#3b82f6',data:hist.tps,width:2,label:'decode last'},\n"
"    {color:'#22c55e',data:hist.tpsAvg,width:1.4,label:'decode avg'},\n"
"    {color:'#a855f7',data:hist.pre,width:1.4,label:'prompt processing'}],v=>v.toFixed(1)+' tok/s');\n"
"  legend($('tpsLegend'),[{color:'#3b82f6',label:'decode, last sample'},{color:'#22c55e',label:'decode, avg of a 20-sample window'},{color:'#1d4ed8',label:'decode min/max of that window'},{color:'#a855f7',label:'prompt processing'}]);\n"
"  const H=stat(t,'host_ms'),HR=stat(t,'host_read_ms'),D=stat(t,'device_wait_ms'),DU=stat(t,'device_upload_ms'),DR=stat(t,'device_readback_ms'),DS=stat(t,'device_sync_ms');\n"
"  stack($('costBar'),[\n"
"    {label:'host: expert read + staging',value:HR.average,color:'#3b82f6'},\n"
"    {label:'host: encoding, launches, sampling',value:sub(H,HR).average,color:'#60a5fa'},\n"
"    {label:'device: expert upload',value:DU.average,color:'#f59e0b'},\n"
"    {label:'device: result readback',value:DR.average,color:'#fbbf24'},\n"
"    {label:'device: other synchronisation',value:DS.average,color:'#fcd34d'}],\n"
"    `Average token cost by component (${fmtMs(ptt.average)} total)`,fmtMs);\n"
"  const P=Number(ptt.average)||0;\n"
"  const rows=[['Host: reading and staging routed experts',HR,'part of host'],\n"
"    ['Host: graph encoding, kernel launches, sampling, detokenizing',sub(H,HR),'part of host'],\n"
"    ['Host: the processor working',H,'total'],\n"
"    ['Device: waiting for the expert upload',DU,'part of device'],\n"
"    ['Device: reading results back',DR,'part of device'],\n"
"    ['Device: other synchronisation',DS,'part of device'],\n"
"    ['Device: the processor blocked on the card',D,'total'],\n"
"    ['Total per token (host + device)',stat(t,'per_token_ms'),'total'],\n"
"    ['Weight bytes at the measured link peak',stat(t,'weight_transfer_ms'),'overlaps'],\n"
"    ['Activation bytes at the measured link peak',stat(t,'activation_transfer_ms'),'overlaps']];\n"
"  $('breakdown').innerHTML='<tr><th>Component</th><th>Share of token</th><th>Last</th><th>Min</th><th>Max</th><th>Avg</th></tr>'+\n"
"    rows.map(r=>`<tr><td>${r[0]} <span class='muted'>(${r[2]})</span></td><td>${r[2]==='overlaps'?'-':pctstr(r[1].average,P)}</td><td>${fmtMs(r[1].last)}</td><td>${fmtMs(r[1].min)}</td><td>${fmtMs(r[1].max)}</td><td>${fmtMs(r[1].average)}</td></tr>`).join('');\n"
"  const STEPS=d.steps||[];\n"
"  const escTxt=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));\n"
"  const advEl=$('advSteps'),adv=advEl&&advEl.checked;\n"
"  const catName={host:'Host - CPU work (not waiting)',device:'Device - GPU busy',transfer:'Transfer - bytes on a link',wait:'Wait - CPU blocked on the device'};\n"
"  const vis=STEPS.filter(s=>adv||Number(s.ms_per_token)>=0.5||Number(s.pct_of_token)>=1);\n"
"  let sh='<tr><th>Step</th><th>ms/token</th><th>% of token</th><th>calls/token</th><th>bytes/token</th><th>rate</th></tr>';\n"
"  for(const c of ['host','device','transfer','wait']){\n"
"    const rs=vis.filter(s=>s.category===c);\n"
"    if(!rs.length)continue;\n"
"    sh+=`<tr><td colspan='6' class='muted'>${catName[c]||c}</td></tr>`;\n"
"    for(const s of rs) sh+=`<tr title=\"${escTxt(s.hint)}\"><td>${escTxt(s.label)} <span class='muted'>${escTxt(s.id)}</span></td><td><b>${Number(s.ms_per_token).toFixed(2)}</b></td><td>${Number(s.pct_of_token).toFixed(1)}%</td><td>${Number(s.calls_per_token).toFixed(1)}</td><td>${Number(s.bytes_per_token)>0?fmtB(Number(s.bytes_per_token)):'-'}</td><td>${Number(s.gb_per_s)>0?Number(s.gb_per_s).toFixed(2)+' GB/s':'-'}</td></tr>`;\n"
"  }\n"
"  if(!STEPS.length) sh+=\"<tr><td colspan='6' class='muted'>no per-step samples yet</td></tr>\";\n"
"  $('steps').innerHTML=sh;\n"
"  const TOK=d.tokens||[],EV=d.events||[];\n"
"  let tl='<table><tr><th>Token</th><th>ms</th><th>Every device (utilization)</th><th>Experts: VRAM / RAM / SSD / CPU</th><th>compute ops</th><th>events</th></tr>';\n"
"  tl+=TOK.map(r=>`<tr><td>#${r.token}</td><td><b>${Number(r.ms).toFixed(1)}</b></td><td>${(r.devices||[]).map(x=>`${escTxt(x.name.replace('NVIDIA GeForce ',''))} <b>${x.util}%</b> <span class='muted'>${fmtB(x.mem_used)}</span>`).join(' &middot; ')}</td><td>${r.pool} / ${r.arena} / ${r.ssd} / ${r.cpu}</td><td>${r.compute}</td><td>${r.events}</td></tr>`).join('');\n"
"  if(!TOK.length) tl+='<tr><td colspan=6 class=muted>no token samples yet</td></tr>';\n"
"  $('tokList').innerHTML=tl;\n"
"  const advEv=$('advEvents')&&$('advEvents').checked;\n"
"  const evs=advEv?EV:EV.slice(-60);\n"
"  let el='<table><tr><th>time &micro;s</th><th>event</th><th>layer</th><th>expert</th><th>GPU tier</th><th>bytes</th></tr>';\n"
"  el+=evs.map(e=>`<tr><td>${e.us}</td><td>${escTxt(e.k)}</td><td>${e.layer}</td><td>${e.expert?e.expert:'-'}</td><td>${e.dev}</td><td>${e.bytes?fmtB(e.bytes):'-'}</td></tr>`).join('');\n"
"  if(!EV.length) el+='<tr><td colspan=6 class=muted>no events yet</td></tr>';\n"
"  $('tokEvents').innerHTML=el;\n"
"  const e=d.expert_distribution||{};\n"
"  const etot=(e.total||0)||1;\n"
"  const big=(t,v,share)=>`<div class='card'><div class='muted'>${t}</div><div class='hero'>${v}</div><div class='muted'>${share}</div></div>`;\n"
"  $('experts').innerHTML=`<div class='cards'>`+\n"
"    big('Total routed experts',e.total||0,'100%')+\n"
"    big('Resident on devices (VRAM)',e.on_gpu||0,pctstr(e.on_gpu||0,etot))+\n"
"    big('Held in host RAM',e.in_host_ram||0,pctstr(e.in_host_ram||0,etot))+\n"
"    big('Streamed from SSD',e.on_ssd||0,pctstr(e.on_ssd||0,etot))+\n"
"    `</div><div id='expBar'></div><div class='legend'><span>Each routed expert is ${fmtB(e.bytes_per_expert)}</span>`+\n"
"    `<span>Device resident set: ${fmtB((e.on_gpu||0)*(e.bytes_per_expert||0))}</span>`+\n"
"    `<span>Host set: ${fmtB((e.in_host_ram||0)*(e.bytes_per_expert||0))}</span></div>`;\n"
"  stack($('expBar'),[\n"
"    {label:'resident on devices',value:(e.on_gpu||0)*(e.bytes_per_expert||0),color:'#3b82f6'},\n"
"    {label:'held in host RAM',value:(e.in_host_ram||0)*(e.bytes_per_expert||0),color:'#22c55e'},\n"
"    {label:'streamed from SSD',value:(e.on_ssd||0)*(e.bytes_per_expert||0),color:'#64748b'}],'Expert bytes by tier');\n"
"  const now=Date.now(),dt=prevTime?(now-prevTime)/1000:0;\n"
"  const ssdNow=Number(h.ssd_read_bytes)||0;\n"
"  const srate=(dt>0&&prevSsd&&ssdNow>=prevSsd)?(ssdNow-prevSsd)/dt:0;\n"
"  if(srate>ssdPeak)ssdPeak=srate;\n"
"  const pinned=Number(h.arena_used_bytes)||0;\n"
"  const cache=(Number(h.arena_capacity_bytes)>0)?0:(Number(h.ram_bytes)||0);\n"
"  const totalRam=Number(h.ram_total_bytes)||0;\n"
"  stack($('storage'),[\n"
"    {label:'pinned expert arena',value:pinned,color:'#22c55e'},\n"
"    {label:'page-cache expert budget',value:cache,color:'#3b82f6'},\n"
"    {label:'free for kernel and other apps',value:Math.max(0,totalRam-pinned-cache),color:'#64748b'}],\n"
"    `Host memory: ${fmtB(totalRam)} total`);\n"
"  $('storage').innerHTML+=`<div class='comp'>`+\n"
"    `<div><span>Model data read now</span><b>${fmtRate(srate)}</b></div>`+\n"
"    `<div><span>Model data read, peak seen</span><b>${fmtRate(ssdPeak)}</b></div>`+\n"
"    `<div><span>Model data read, total</span><b>${fmtB(h.ssd_read_bytes)}</b></div>`+\n"
"    `<div><span>Model data written</span><b>none during inference</b></div>`+\n"
"    `<div><span>Model size on storage</span><b>${fmtB(h.model_bytes)}</b></div>`+\n"
"    `<div><span>Experts held in host RAM</span><b>${h.ram_experts}</b></div>`+\n"
"    `<div><span>Host RAM held by them</span><b>${fmtB(h.ram_bytes)}</b></div>`+\n"
"    (Number(h.arena_capacity_bytes)>0?`<div><span>Pinned arena</span><b>${fmtB(h.arena_used_bytes)} of ${fmtB(h.arena_capacity_bytes)}</b></div>`:'')+\n"
"    (Number(h.arena_capacity_bytes)>0?`<div><span>Arena fetches served / read from storage</span><b>${h.arena_hits} / ${h.arena_misses}</b></div>`:'')+\n"
"    (Number(h.arena_capacity_bytes)>0?`<div><span>Arena admissions / evictions</span><b>${h.arena_admissions} / ${h.arena_evictions}</b></div>`:'')+\n"
"    (Number(h.arena_capacity_bytes)>0?`<div><span>Fetches re-reading an evicted expert</span><b>${h.arena_evicted_fetches}</b></div>`:'')+\n"
"    `</div>`;\n"
"  const pcSeries=[];\n"
"  (d.devices||[]).forEach(dev=>{\n"
"    const into=rate(dev,dt,'pcie_tx_bytes'),out=rate(dev,dt,'pcie_rx_bytes');\n"
"    const key='d'+dev.device_id;hist.pcie[key]=hist.pcie[key]||{tx:[],rx:[]};\n"
"    push(hist.pcie[key].tx,into);push(hist.pcie[key].rx,out);\n"
"    pcSeries.push({color:dev.tier===0?'#3b82f6':'#f59e0b',data:hist.pcie[key].tx,width:1.8,label:`tier ${dev.tier} into device`});\n"
"    pcSeries.push({color:dev.tier===0?'#93c5fd':'#fcd34d',data:hist.pcie[key].rx,width:1.2,alpha:.9,label:`tier ${dev.tier} out of device`});\n"
"  });\n"
"  chart($('pcieChart'),pcSeries,fmtRate);\n"
"  legend($('pcieLegend'),pcSeries.map(s=>({color:s.color,label:s.label})));\n"
"  $('devices').innerHTML=(d.devices||[]).map(dev=>deviceCard(dev,dt)).join('');\n"
"  (d.devices||[]).forEach(dev=>prevDev[dev.device_id]=dev);\n"
"  prevHost=d;prevTime=now;prevSsd=ssdNow;\n"
"}\n"
"function rate(dev,dt,dir){const p=prevDev[dev.device_id];if(!p||dt<=0)return 0;const a=Number(dev[dir])||0,b=Number(p[dir])||0;return a>=b?(a-b)/dt:0;}\n"
"function deviceCard(dev,dt){\n"
"  const used=Number(dev.vram_used_bytes)||0,total=Number(dev.vram_total_bytes)||1;\n"
"  const comp=[['dense weights',dev.dense_bytes,'#3b82f6'],['weight arena',dev.arena_bytes,'#2563eb'],['key/value cache',dev.kv_bytes,'#8b5cf6'],['resident experts',dev.expert_bytes,'#22c55e'],['graph scratch',dev.scratch_bytes,'#f59e0b'],['temporary buffers',dev.temp_bytes,'#fb923c'],['tensor allocations',dev.tensor_bytes,'#14b8a6'],['reserved pool',dev.pool_bytes,'#64748b'],['driver and desktop',dev.other_bytes,'#475569']].filter(x=>Number(x[1])>0);\n"
"  const vramStack=`<div class='stack'>`+comp.map(x=>`<i style='width:${(100*Number(x[1])/total).toFixed(2)}%;background:${x[2]}'></i>`).join('')+`</div>`;\n"
"  const vramLegend=`<div class='legend'>`+comp.map(x=>`<span><b style='color:${x[2]}'>&#9632;</b> ${x[0]} <b>${pctstr(x[1],total)}</b> <span class='muted'>${fmtB(x[1])}</span></span>`).join('')+`</div>`;\n"
"  const into=rate(dev,dt,'pcie_tx_bytes'),out=rate(dev,dt,'pcie_rx_bytes');\n"
"  const wrate=rate(dev,dt,'weight_bytes'),arate=rate(dev,dt,'activation_bytes');\n"
"  const peak=Number(dev.pcie_peak_bytes_per_s)||0;\n"
"  const util=peak>0?pctstr(Math.max(into,out),peak)+' of measured peak':'not measured';\n"
"  const spec=[['Name',dev.name||('device '+dev.device_id)],['Compute capability',dev.tensor_gen?Math.floor(dev.tensor_gen/10)+'.'+(dev.tensor_gen%10):'-'],['Streaming multiprocessors',dev.sm_count||'-'],['VRAM bandwidth',dev.mem_bw_bytes_per_s?fmtGBps(dev.mem_bw_bytes_per_s):'-'],['fp16 tensor throughput (peak)',dev.compute_flops_fp16?fmtTflops(dev.compute_flops_fp16):'-'],['PCIe link',dev.pcie_gen?('gen '+dev.pcie_gen+' x '+dev.pcie_width):'not exposed by CUDA'],['Measured link peak',peak?fmtGBps(peak):'-'],['Peer to peer',dev.nvlink?'NVLink':(dev.p2p?'available':'not available')]];\n"
"  const specHtml=spec.map(x=>`<tr><td>${x[0]}</td><td>${x[1]}</td></tr>`).join('');\n"
"  return `<div class='card'><div class='devhead'><div><b>Tier ${dev.tier}</b> <span class='muted'>device ${dev.device_id}</span></div><span class='tag'>${role(dev.tier)}</span></div>`+\n"
"    `<div class='muted'>VRAM ${fmtB(used)} of ${fmtB(total)} used (${pctstr(used,total)})</div>`+vramStack+vramLegend+\n"
"    `<table style='margin-top:12px'><tr><th>Specification</th><th></th></tr>`+specHtml+`</table>`+\n"
"    `<table style='margin-top:12px'><tr><th>Data movement</th><th>Now</th><th>Total since start</th></tr>`+\n"
"    `<tr><td>Weights read into the device</td><td>${fmtRate(wrate)}</td><td>${fmtB(dev.weight_bytes)}</td></tr>`+\n"
"    `<tr><td>Activations moved onto the device</td><td>${fmtRate(arate)}</td><td>${fmtB(dev.activation_bytes)}</td></tr>`+\n"
"    `<tr><td>Total read in over PCI Express</td><td>${fmtRate(into)}</td><td>${fmtB(dev.pcie_tx_bytes)}</td></tr>`+\n"
"    `<tr><td>Total written out over PCI Express</td><td>${fmtRate(out)}</td><td>${fmtB(dev.pcie_rx_bytes)}</td></tr>`+\n"
"    `<tr><td>Link utilisation</td><td>${util}</td><td></td></tr>`+\n"
"    `</table>`+\n"
"    (dev.expert_count>0?`<div class='legend'><span>${dev.expert_count} resident experts at ${fmtB(dev.expert_size_bytes)} each</span><span>${dev.expert_type||''}</span></div>`:'')+\n"
"    `</div>`;\n"
"}\n"
"async function tick(){try{const r=await fetch('/stats.json',{cache:'no-store'});render(await r.json());}catch(err){$('status').textContent='dashboard lost connection to the engine';}}\n"
"tick();setInterval(tick,300);\n"
"</script>\n"
"</body>\n"
"</html>\n";

/* ---------------------------------------------------------------- server -- */

static int             g_fd = -1;
static volatile int    g_running;
static pthread_t       g_thread;
static char            g_url[64];

static void send_all(int fd, const char *buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        const ssize_t w = send(fd, buf + off, n - off, MSG_NOSIGNAL);
        if (w <= 0) return;
        off += (size_t)w;
    }
}

static void http_reply(int fd, const char *status, const char *ctype,
                       const char *body, size_t blen) {
    char head[256];
    const int hn = snprintf(head, sizeof(head),
            "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
            status, ctype, blen);
    if (hn > 0) send_all(fd, head, (size_t)hn);
    if (blen) send_all(fd, body, blen);
}

/* Integer query parameter ("?seq=12"), or `def` when absent or unparsable.
 * Path and query share `path`; the comparison below is length-bounded so a
 * query never changes which route is taken. */
static long query_long(const char *path, const char *key, long def) {
    char pat[32];
    snprintf(pat, sizeof(pat), "%s=", key);
    const char *q = strchr(path, '?');
    if (!q) return def;
    const char *p = q + 1;
    const size_t plen = strlen(pat);
    while (p && *p) {
        if (!strncmp(p, pat, plen)) {
            char *end = NULL;
            const long v = strtol(p + plen, &end, 10);
            return end == p + plen ? def : v;
        }
        p = strchr(p, '&');
        if (p) p++;
    }
    return def;
}

/* Record kind filter: "decode" (default), "prefill", "chunk", "all". */
static int query_kind(const char *path) {
    const char *q = strchr(path, '?');
    if (!q) return DS4_PREC_DECODE_TOKEN;
    if (strstr(q, "kind=all")) return -1;
    if (strstr(q, "kind=prefill")) return DS4_PREC_PREFILL_CHUNK;
    if (strstr(q, "kind=chunk")) return DS4_PREC_PREFILL_CHUNK;
    if (strstr(q, "kind=decode")) return DS4_PREC_DECODE_TOKEN;
    return DS4_PREC_DECODE_TOKEN;
}

static int route_is(const char *path, const char *route) {
    const size_t n = strlen(route);
    if (strncmp(path, route, n) != 0) return 0;
    return path[n] == '\0' || path[n] == '?';
}

static void handle_conn(int fd) {
    char req[4096];
    const ssize_t n = recv(fd, req, sizeof(req) - 1u, 0);
    if (n <= 0) return;
    req[n] = '\0';
    char method[16] = {0}, path[1024] = {0};
    if (sscanf(req, "%15s %1023s", method, path) != 2) return;
    if (strcmp(method, "GET") != 0 && strcmp(method, "HEAD") != 0) {
        static const char m[] = "method not allowed\n";
        http_reply(fd, "405 Method Not Allowed", "text/plain; charset=utf-8",
                   m, sizeof(m) - 1u);
        return;
    }
    if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) {
        http_reply(fd, "200 OK", "text/html; charset=utf-8",
                   DASHBOARD_HTML, sizeof(DASHBOARD_HTML) - 1u);
        return;
    }
    if (strcmp(path, "/stats.json") == 0) {
        static char *json;
        static size_t cap;
        if (!json) { cap = 1024u * 1024u; json = malloc(cap); }
        if (!json) return;
        const size_t len = ds4_profiler_json(json, cap);
        http_reply(fd, "200 OK", "application/json; charset=utf-8", json, len);
        return;
    }
    /* Profiler surface for other processes. The text routes exist because a
     * caller that has to *read* the answer (a model, a shell script) pays for
     * every JSON brace; the JSON routes are for programs that parse it. */
    if (route_is(path, "/profiler/records") ||
        route_is(path, "/profiler/records.json")) {
        static char *json;
        static size_t cap;
        if (!json) { cap = 256u * 1024u; json = malloc(cap); }
        if (!json) return;
        const size_t len = ds4_profiler_records_dump_json(
            json, cap, query_kind(path), (int)query_long(path, "limit", 16));
        http_reply(fd, "200 OK", "application/json; charset=utf-8", json, len);
        return;
    }
    if (route_is(path, "/profiler/records.txt")) {
        static char *txt;
        static size_t cap;
        if (!txt) { cap = 256u * 1024u; txt = malloc(cap); }
        if (!txt) return;
        const size_t len = ds4_profiler_records_dump(
            txt, cap, query_kind(path), (int)query_long(path, "limit", 16));
        http_reply(fd, "200 OK", "text/plain; charset=utf-8", txt, len);
        return;
    }
    if (route_is(path, "/profiler/record") ||
        route_is(path, "/profiler/record.json") ||
        route_is(path, "/profiler/record.txt")) {
        static char *buf;
        static size_t cap;
        if (!buf) { cap = DS4_PROF_DUMP_CAP; buf = malloc(cap); }
        if (!buf) return;
        const long seq = query_long(path, "seq", -1);
        if (seq < 0) {
            static const char bad[] =
                "{\"error\":\"seq is required, see /profiler/records\"}\n";
            http_reply(fd, "400 Bad Request", "application/json; charset=utf-8",
                       bad, sizeof(bad) - 1u);
            return;
        }
        const int text = route_is(path, "/profiler/record.txt");
        const size_t len = text
            ? ds4_profiler_record_dump((uint64_t)seq, buf, cap)
            : ds4_profiler_record_dump_json((uint64_t)seq, buf, cap);
        http_reply(fd, "200 OK",
                   text ? "text/plain; charset=utf-8"
                        : "application/json; charset=utf-8", buf, len);
        return;
    }
    if (route_is(path, "/profiler/report") || route_is(path, "/profiler/report.txt")) {
        static char *txt;
        static size_t cap;
        if (!txt) { cap = 4u * DS4_PROF_DUMP_CAP; txt = malloc(cap); }
        if (!txt) return;
        const size_t len = ds4_profiler_report(
            txt, cap, query_kind(path), (int)query_long(path, "limit", 8));
        http_reply(fd, "200 OK", "text/plain; charset=utf-8", txt, len);
        return;
    }
    static const char nf[] = "not found\n";
    http_reply(fd, "404 Not Found", "text/plain; charset=utf-8", nf, sizeof(nf) - 1u);
}

static void *server_main(void *arg) {
    (void)arg;
    while (g_running) {
        struct pollfd p = { g_fd, POLLIN, 0 };
        const int rc = poll(&p, 1, 200);
        if (rc <= 0) continue;
        const int c = accept(g_fd, NULL, NULL);
        if (c < 0) continue;
        handle_conn(c);
        close(c);
    }
    return NULL;
}

/* Launch the default URL handler without leaving a zombie behind. */
static void open_url(const char *url) {
    const pid_t pid = fork();
    if (pid == 0) {
        (void)setsid();
        const pid_t p2 = fork();
        if (p2 == 0) {
            (void)execlp("xdg-open", "xdg-open", url, (char *)NULL);
            (void)execlp("gio", "gio", "open", url, (char *)NULL);
            _exit(127);
        }
        _exit(0);
    }
    if (pid > 0) (void)waitpid(pid, NULL, 0);
}

int ds4_dashboard_active(void) { return g_running; }
const char *ds4_dashboard_url(void) { return g_running ? g_url : ""; }

/* Print the addresses another machine on the network can reach this on. */
static void print_lan_urls(int port) {
    struct ifaddrs *ifa = NULL;
    if (getifaddrs(&ifa) != 0) return;
    for (struct ifaddrs *p = ifa; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        const struct sockaddr_in *s = (const struct sockaddr_in *)p->ifa_addr;
        if ((ntohl(s->sin_addr.s_addr) >> 24) == 127u) continue;  /* loopback */
        char host[INET_ADDRSTRLEN] = {0};
        if (!inet_ntop(AF_INET, &s->sin_addr, host, sizeof(host))) continue;
        fprintf(stderr, "ds4: engine dashboard on http://%s:%d/ (%s)\n",
                host, port, p->ifa_name);
    }
    freeifaddrs(ifa);
}

int ds4_dashboard_start(const char *bind_addr, int port, int open_browser) {
    if (g_running) return 1;
    const char *env = getenv("DS4_DASHBOARD");
    if (env && (env[0] == '0' || strcmp(env, "off") == 0)) return 0;
    if (!bind_addr || !bind_addr[0]) bind_addr = getenv("DS4_DASHBOARD_BIND");
    if (!bind_addr || !bind_addr[0]) bind_addr = "127.0.0.1";
    const int loopback = strcmp(bind_addr, "127.0.0.1") == 0 ||
                         strcmp(bind_addr, "localhost") == 0;
    /* A loopback dashboard is only useful where someone can open it, so a
     * headless/redirected run gets none. An explicit non-loopback bind is the
     * operator asking for network access, so it starts without a terminal. */
    if (loopback && !isatty(STDOUT_FILENO)) return 0;
    if (!loopback) {
        fprintf(stderr, "ds4: engine dashboard has no authentication; it is "
                        "being exposed on %s, use it only on a trusted network\n",
                bind_addr);
    }
    if (port <= 0) port = DS4_DASH_DEFAULT_PORT;

    int fd = -1, chosen = 0;
    for (int p = port; p < port + DS4_DASH_PORT_TRIES; p++) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return 0;
        int one = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        struct sockaddr_in a;
        memset(&a, 0, sizeof(a));
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = loopback ? htonl(INADDR_LOOPBACK)
                                     : inet_addr(bind_addr);
        a.sin_port = htons((uint16_t)p);
        if (bind(fd, (struct sockaddr *)&a, sizeof(a)) == 0 &&
            listen(fd, 8) == 0) {
            chosen = p;
            break;
        }
        close(fd);
        fd = -1;
    }
    if (fd < 0) {
        fprintf(stderr, "ds4: engine dashboard: no free port near %d\n", port);
        return 0;
    }

    ds4_gpu_profiler_init();  /* measure link peaks for byte -> time */
    g_fd = fd;
    snprintf(g_url, sizeof(g_url), "http://%s:%d/",
             loopback ? "127.0.0.1" : bind_addr, chosen);
    g_running = 1;
    if (pthread_create(&g_thread, NULL, server_main, NULL) != 0) {
        g_running = 0;
        close(fd);
        g_fd = -1;
        return 0;
    }
    fprintf(stderr, "ds4: engine dashboard: %s\n", g_url);
    if (!loopback) print_lan_urls(chosen);
    if (open_browser && loopback) open_url(g_url);
    return 1;
}

void ds4_dashboard_stop(void) {
    if (!g_running) return;
    g_running = 0;
    pthread_join(g_thread, NULL);   /* poll() timeout bounds the wait */
    if (g_fd >= 0) close(g_fd);
    g_fd = -1;
    fprintf(stderr, "ds4: engine dashboard stopped\n");
}

#endif /* !_WIN32 */
