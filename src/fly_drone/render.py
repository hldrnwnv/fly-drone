"""Write a standalone, interactive replay of the virtual flight."""

import json
from pathlib import Path


def write_replay(path: Path, runs: dict, metrics: dict) -> None:
    payload = json.dumps({"runs": runs, "metrics": metrics}, ensure_ascii=False).replace("</", "<\\/")
    page = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fly Drone — replay</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#eaf2f8;background:#101a28}
body{max-width:1080px;margin:0 auto;padding:24px}h1{margin:0 0 5px;font-size:30px}p{color:#aabdd0}
.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(240px,1fr);gap:18px}
canvas,.panel{background:#17273b;border:1px solid #30455b;border-radius:14px}
canvas{width:100%;aspect-ratio:1.3}.panel{padding:20px}.metric{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #30455b}
.metric strong{font-variant-numeric:tabular-nums}label{display:block;margin:16px 0 6px}select,input,button{accent-color:#4ee1b1}
select,button{background:#233a52;color:#fff;border:1px solid #4b6983;border-radius:8px;padding:9px 12px;cursor:pointer}
input[type=range]{width:100%}.small{font-size:13px;line-height:1.5}code{color:#7ee8bd}
@media(max-width:760px){.layout{grid-template-columns:1fr}}
</style></head><body><h1>Мозг дрозофилы → виртуальный дрон</h1>
<p>Цель: долететь до зелёной точки. Высота и скорость фиксированы; нейронный выход задаёт поворот.</p>
<div class="layout"><div><canvas id="map" width="780" height="600"></canvas>
<label for="timeline">Время <span id="time">0.0</span> с</label><input id="timeline" type="range" min="0" value="0" step="1">
<button id="play">▶ Воспроизвести</button></div><div class="panel">
<label for="run">Контроллер</label><select id="run"></select><div id="stats"></div>
<p class="small">Голубая линия — путь дрона. Зелёный круг — цель. Пунктир — идеальная прямая. Нейронные связи не обучались: обучался только линейный декодер нисходящей активности.</p>
<p class="small">Данные: MaleCNS v1.0 через <code>flybrain 0.1.0</code>. Сенсорный вход задаётся синтетически по направлению на цель.</p>
</div></div><script>
const data=__DATA__;const runs=data.runs, canvas=document.getElementById('map'),ctx=canvas.getContext('2d');
const select=document.getElementById('run'),slider=document.getElementById('timeline'),time=document.getElementById('time'),stats=document.getElementById('stats');
Object.keys(runs).forEach(name=>{let o=document.createElement('option');o.value=name;o.textContent=name;select.appendChild(o)});
let timer=null;function stop(){if(timer){clearInterval(timer);timer=null}document.getElementById('play').textContent='▶ Воспроизвести'}
function display(){const r=runs[select.value],step=Number(slider.value),point=step?r.trace[step-1]:{x:0,y:0,yaw:0,t:0,command:0,distance:Math.hypot(...r.target)};
 const all=[{x:0,y:0},...r.trace,{x:r.target[0],y:r.target[1]}],xs=all.map(p=>p.x),ys=all.map(p=>p.y);
 let minx=Math.min(...xs)-1,maxx=Math.max(...xs)+1,miny=Math.min(...ys)-1,maxy=Math.max(...ys)+1;
 let scale=Math.min((canvas.width-90)/(maxx-minx),(canvas.height-90)/(maxy-miny));
 let xy=(x,y)=>[45+(x-minx)*scale,canvas.height-45-(y-miny)*scale];
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#17273b';ctx.fillRect(0,0,canvas.width,canvas.height);
 ctx.strokeStyle='#30455b';ctx.lineWidth=1;for(let x=Math.ceil(minx);x<=maxx;x++){let a=xy(x,miny),b=xy(x,maxy);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}for(let y=Math.ceil(miny);y<=maxy;y++){let a=xy(minx,y),b=xy(maxx,y);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
 ctx.strokeStyle='#6a8297';ctx.setLineDash([6,7]);ctx.beginPath();ctx.moveTo(...xy(0,0));ctx.lineTo(...xy(...r.target));ctx.stroke();ctx.setLineDash([]);
 ctx.strokeStyle='#63caff';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(...xy(0,0));for(let i=0;i<step;i++)ctx.lineTo(...xy(r.trace[i].x,r.trace[i].y));ctx.stroke();
 const goal=xy(...r.target);ctx.fillStyle='#4ee1b1';ctx.beginPath();ctx.arc(...goal,12,0,2*Math.PI);ctx.fill();
 const p=xy(point.x,point.y);ctx.save();ctx.translate(...p);ctx.rotate(-point.yaw);ctx.fillStyle='#ffcf73';ctx.beginPath();ctx.moveTo(19,0);ctx.lineTo(-13,-10);ctx.lineTo(-8,0);ctx.lineTo(-13,10);ctx.closePath();ctx.fill();ctx.restore();
 const group=r.target[1]>0?'цель слева':'цель справа',series=data.metrics.flights[group].brain;
 time.textContent=point.t.toFixed(1);stats.innerHTML=`<div class="metric"><span>Цель достигнута</span><strong>${r.reached?'да':'нет'}</strong></div><div class="metric"><span>Итоговая дистанция</span><strong>${r.final_distance.toFixed(2)} м</strong></div><div class="metric"><span>Шагов</span><strong>${r.steps}</strong></div><div class="metric"><span>Текущая команда</span><strong>${point.command.toFixed(2)}</strong></div><div class="metric"><span>Успехов модели</span><strong>${series.successes}/${series.trials}</strong></div><div class="metric"><span>Точность на отложенных примерах</span><strong>${(100*data.metrics.validation.direction_accuracy).toFixed(0)}%</strong></div>`;
}
select.onchange=()=>{stop();slider.max=runs[select.value].trace.length;slider.value=0;display()};slider.oninput=display;
document.getElementById('play').onclick=()=>{if(timer){stop();return}document.getElementById('play').textContent='❚❚ Пауза';timer=setInterval(()=>{if(Number(slider.value)>=Number(slider.max)){stop();return}slider.value=Number(slider.value)+1;display()},80)};
select.dispatchEvent(new Event('change'));
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page.replace("__DATA__", payload), encoding="utf-8")
