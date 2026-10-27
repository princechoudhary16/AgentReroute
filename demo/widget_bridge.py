"""Same-origin presentation bridge for the two existing interactive tools."""
import json


def embed(page, widget):
    if widget not in ('system', 'compare'):
        raise ValueError('Unknown studio widget')
    style = '''<style>
html,body{min-height:0;height:auto;background:#090e11;overflow:hidden}
.studio-shell{max-width:none;padding:24px;margin:0;min-height:0}
.app-nav,.studio-hero,.studio-footer,.footer{display:none!important}
.story-motion-paused *{animation:none!important;transition:none!important}
.canvas{touch-action:pan-y pinch-zoom}
@media(max-width:760px){.studio-shell{padding:14px 10px}}
</style>'''
    script = r'''<script>
window.STUDIO_EMBEDDED=true;
(function(){
 const widget=__WIDGET__, origin=location.origin;
 window.storyGuide=detail=>{
  if(parent===window)return;
  parent.postMessage({type:'agentreroute:guide',widget,...detail,
   runId:new URLSearchParams(location.search).get('run')},origin);
 };
 addEventListener('message',e=>{
  if(e.origin!==origin||e.source!==parent||e.data?.type!=='agentreroute:motion'||typeof e.data.paused!=='boolean')return;
  window.STUDIO_REDUCED_MOTION=e.data.paused;
  document.documentElement.classList.toggle('story-motion-paused',e.data.paused);
 });
 let lastHeight=0,queued=false;
 function measure(){
  queued=false;
  const root=document.querySelector('main');
  if(!root)return;
  const height=Math.ceil(root.getBoundingClientRect().height)+2;
  if(height!==lastHeight){lastHeight=height;parent.postMessage({type:'agentreroute:resize',widget,height},origin);}
 }
 function queue(){if(!queued){queued=true;requestAnimationFrame(measure);}}
 document.addEventListener('DOMContentLoaded',()=>{
  const hint=document.querySelector('.canvas-legend small');
  if(hint)hint.textContent='Use + / − to zoom · Scroll to explore';
  new ResizeObserver(queue).observe(document.querySelector('main'));
  window.addEventListener('resize',queue);queue();
 });
})();
</script>'''.replace('__WIDGET__', json.dumps(widget))
    return page.replace('</head>', style + script + '</head>')
