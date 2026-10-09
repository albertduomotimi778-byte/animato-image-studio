import { normalizeSpec, geometryFor, type CharacterSpec, type CssRig, type ArmMeta } from './cssCharacter';

/** Runtime rig builder for AI-rigged raster characters. The stored rig contains
 * asset:<sha1> references; the renderer/preview supplies an assetBaseUrl. */
export function buildAiRig(input: any, assetBaseUrl = ''): CssRig {
  const rig = input?.rig && input.rig.characters ? input.rig : input;
  const sourceSpec = input?.kind === 'ai-rig' ? input : rig?.spec || {};
  const gender = sourceSpec?.gender === 'male' ? 'male' : 'female';
  // A portrait image has no legs to show: honour what the pipeline detected.
  const fullBody = rig?.detected?.fullBody === false ? false : !!sourceSpec?.fullBody;
  let spec: any;
  try {
    spec = normalizeSpec({
      ...sourceSpec,
      kind: undefined,
      name: sourceSpec?.name || rig?.characters?.[0]?.name || 'AI Character',
      gender,
      fullBody,
      style: 'classic',
      hair: { ...(sourceSpec?.hair || {}), style: sourceSpec?.hair?.style || 'short' }
    }, gender);
  } catch {
    spec = normalizeSpec({ name: sourceSpec?.name || 'AI Character', gender, fullBody, style: 'classic' }, gender);
  }
  const resolve = (u: any): string | null => {
    if (!u) return null;
    if (typeof u !== 'string') return null;
    if (u.startsWith('asset:')) {
      const id = u.slice(6);
      const base = String(assetBaseUrl || sourceSpec?.assetBaseUrl || rig?.assetBaseUrl || '').replace(/\/+$/, '');
      return base ? `${base}/api/automation/assets/${id}` : u;
    }
    return u;
  };
  const chars = Array.isArray(rig?.characters) ? rig.characters : [];
  const first = chars[0] || { id: 'presenter', name: spec.name || 'AI Character', composition: {} };
  const composition: Record<string, any> = {};
  for (const [id, p0] of Object.entries(first.composition || {})) {
    const p: any = p0 || {};
    composition[id] = {
      id,
      label: p.label || id,
      tags: Array.isArray(p.tags) ? p.tags : [],
      imageUrl: resolve(p.imageUrl),
      originalImageUrl: resolve(p.originalImageUrl),
      transform: {
        x: Number(p.transform?.x || 0), y: Number(p.transform?.y || 0),
        rotation: Number(p.transform?.rotation || 0),
        scaleX: Number(p.transform?.scaleX ?? 1), scaleY: Number(p.transform?.scaleY ?? 1),
        anchorX: Number(p.transform?.anchorX ?? 50), anchorY: Number(p.transform?.anchorY ?? 50),
        ...(p.transform?.flipX ? { flipX: true } : {}), ...(p.transform?.flipY ? { flipY: true } : {})
      },
      baseTransform: p.baseTransform || p.transform || undefined,
      width: Number(p.width || 100), height: Number(p.height || 100),
      zIndex: Number(p.zIndex || 0), parentId: p.parentId || null,
      children: Array.isArray(p.children) ? p.children.slice() : [],
      isGroup: !!p.isGroup, isIndependent: p.isIndependent !== false, isVisible: p.isVisible !== false,
      opacity: Number(p.opacity ?? 1), rigType: p.rigType || 'raster'
    };
  }
  // Rigs from the first pipeline (version 1) used raw image-pixel x (0..width) and listed
  // children back→front, while the engine centres x = 0 and paints the FIRST child on top.
  // Normalise them here so previously saved characters render correctly too.
  const legacy = Number(rig?.version || 1) < 2 || rig?.origin !== 'center';
  const shiftX = legacy && rig?.canvas?.width ? Number(rig.canvas.width) / 2 : 0;
  if (legacy) {
    for (const p of Object.values(composition) as any[]) {
      if (shiftX && p.id !== 'root') p.transform.x -= shiftX;
      if ((p.id === 'root' || p.id === 'headGroup') && Array.isArray(p.children)) p.children.reverse();
    }
  }
  if (!composition.root) composition.root = { id:'root', label:'AI Character', imageUrl:null, transform:{x:0,y:0,rotation:0,scaleX:1,scaleY:1,anchorX:50,anchorY:50}, zIndex:0, tags:[], parentId:null, children:[], isGroup:true, isIndependent:false, isVisible:true };
  if (!composition.headGroup) {
    const face = composition.face || composition.head || Object.values(composition).find((p:any) => p?.tags?.includes('Head')) as any;
    composition.headGroup = { id:'headGroup', label:'Head Group', imageUrl:null, transform:face?.transform || {x:0,y:0,rotation:0,scaleX:1,scaleY:1,anchorX:50,anchorY:50}, width:face?.width || 250, height:face?.height || 300, zIndex:0, tags:['Head'], parentId:'root', children:[], isGroup:true, isIndependent:false, isVisible:true };
    if (!composition.root.children.includes('headGroup')) composition.root.children.push('headGroup');
  }
  const map = (m: any): Record<string,string> => {
    const out: Record<string,string> = {};
    for (const [k,v] of Object.entries(m || {})) { const r = resolve(v); if (r) out[k] = r; }
    return out;
  };
  const vMap = map(rig?.visemeMap || first.visemeMap || {});
  const mouthSetsRaw = rig?.mouthSets || { neutral: vMap, smile: vMap, happy: vMap, sad: vMap };
  const mouthSets: any = {};
  for (const mood of ['neutral','smile','happy','sad']) mouthSets[mood] = map(mouthSetsRaw[mood] || vMap);
  const eyes = {
    happy: Array.isArray(rig?.eyes?.happy) ? rig.eyes.happy.filter((x:string)=>!!composition[x]) : [],
    open: Array.isArray(rig?.eyes?.open) ? rig.eyes.open.filter((x:string)=>!!composition[x]) : []
  };
  const extras = {
    tears: Array.isArray(rig?.extras?.tears) ? rig.extras.tears.filter((x:string)=>!!composition[x]) : [],
    glow: typeof rig?.extras?.glow === 'string' && composition[rig.extras.glow] ? rig.extras.glow : ''
  };
  const arms: ArmMeta[] = (Array.isArray(rig?.arms) ? rig.arms as ArmMeta[] : [])
    .filter((a: any) => a && composition[a.upper] && composition[a.fore] && a.hands && Object.values(a.hands).every((id: any) => composition[id]))
    .map((a: any) => shiftX ? { ...a, shoulder: { x: Number(a.shoulder?.x || 0) - shiftX, y: Number(a.shoulder?.y || 0) } } : a);
  const fallbackGeometry = (() => {
    const all = Object.values(composition).filter((p:any)=>!p.isGroup && p.imageUrl);
    const bbox = (ids: string[]) => {
      const ps = ids.map(id=>composition[id]).filter(Boolean) as any[];
      if (!ps.length) return null;
      const left = Math.min(...ps.map(p => p.transform.x - p.width/2));
      const right = Math.max(...ps.map(p => p.transform.x + p.width/2));
      const top = Math.min(...ps.map(p => p.transform.y - p.height/2));
      const bottom = Math.max(...ps.map(p => p.transform.y + p.height/2));
      return {left,right,top,bottom};
    };
    const head = bbox(Object.keys(composition).filter(id=>/head|face|eye|brow|hair|ear|nose|mouth|lip/i.test(id)));
    const allb = bbox(Object.keys(composition));
    const top = allb?.top || 0, feet = allb?.bottom || 1000;
    const waist = Number(rig?.geometry?.frame?.waist ?? (top + (feet-top)*0.7));
    const pelvis = Number(rig?.geometry?.frame?.pelvis ?? (top + (feet-top)*0.62));
    const headTop = head?.top ?? top;
    const headBottom = head?.bottom ?? Math.min(feet, headTop + 300);
    const faceW = head ? Math.max(40, head.right-head.left) : 220;
    const faceH = head ? Math.max(50, head.bottom-head.top) : 280;
    const cx = head ? (head.left+head.right)/2 : 0;
    const headC = {x:cx,y:(headTop+headBottom)/2};
    return {
      frame:{top, pelvis, waist, feet}, headC,
      eyeY:Number(rig?.geometry?.eyeY ?? headC.y-faceH*0.14), browY:Number(rig?.geometry?.browY ?? headC.y-faceH*0.25),
      noseY:Number(rig?.geometry?.noseY ?? headC.y+faceH*0.06), mouthY:Number(rig?.geometry?.mouthY ?? headC.y+faceH*0.24), earY:Number(rig?.geometry?.earY ?? headC.y),
      faceW, faceH, eyeDX:faceW*0.21, eyeW:faceW*0.18, eyeH:faceH*0.13, irisR:Math.max(3,faceW*0.04),
      torsoW:Math.max(80,faceW*1.55), torsoH:Math.max(120,waist-headBottom), shoulderX:faceW*0.8, shoulderY:headBottom+faceH*0.22,
      upperW:Math.max(18,faceW*0.16), foreW:Math.max(16,faceW*0.14), upperLen:Math.max(50,faceH*0.7), foreLen:Math.max(45,faceH*0.62), handLen:Math.max(34,faceH*0.28)
    } as any;
  })();
  const geometry: any = { ...(fallbackGeometry as any), ...(rig?.geometry || {}) };
  if (shiftX && rig?.geometry?.headC) geometry.headC = { ...rig.geometry.headC, x: Number(rig.geometry.headC.x || 0) - shiftX };
  if (shiftX && !rig?.geometry?.headC) geometry.headC = { ...fallbackGeometry.headC };
  return {
    kind: 'css', spec,
    characters: [{ id:first.id || 'presenter', name:first.name || spec.name || 'AI Character', composition }],
    visemeMap: Object.keys(vMap).length ? vMap : mouthSets.neutral,
    mouthSets,
    arms,
    eyes,
    extras,
    geometry,
    camera: rig?.camera || {x:0,y:0,scale:1,rotation:0},
    filters: rig?.filters || {}, lights: rig?.lights || [], ambient: Number(rig?.ambient ?? 1),
    aspect: rig?.aspect || {w:9,h:16}
  } as CssRig;
}
