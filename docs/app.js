"use strict";
/*
 * Dashboard del caso BreastDCEDL. El modelo final (15 redes en ONNX) se ejecuta en el navegador.
 *
 * Pipeline, identico al de Python (predecir.py):
 *   1. cada fase es un PNG de 256x256; escala de grises de 8 bits como Pillow convert("L")
 *      (L = (R*19595 + G*38470 + B*7471 + 0x8000) >> 16) y division entre 255;
 *   2. las 15 redes (exportadas en modo evaluacion; el recorte central de 128x128 va dentro) dan sigmoide(logit);
 *   3. media de las 15, recorte a [1e-6, 1 - 1e-6] y calibracion de Platt: p = sigmoide(a*logit(media) + b);
 *   4. pCR = 1 si p >= umbral.
 * Datos: window.DATOS (datos_web.py: metricas, validacion, curvas) y window.WEB (exportar_web.py: modelos, ejemplos).
 * MODO MAQUETA: si la pagina se abre como fichero (o con ?demo), no se cargan los modelos y las predicciones son
 * inventadas (avisado con una franja); todo lo demas usa los datos reales.
 */

const D = window.DATOS;
const W = window.WEB || null;
const DEMO = location.protocol === "file:" || new URLSearchParams(location.search).has("demo");
const LADO = 256, RECORTE = 128, TAM_MAX = 5 * 1024 * 1024, TOLERANCIA = 1e-4;
const FASES = ["PRE", "EARLY", "LATE"];
const SEMILLAS = [42, 1, 2];
const SIN_MOVIMIENTO = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

const estado = {
  sesiones: [], checksumWeb: null, listo: false,
  fases: null,                 // {PRE, EARLY, LATE}: Float32Array de la muestra cargada
  buffers: { PRE: null, EARLY: null, LATE: null }, nombres: { PRE: null, EARLY: null, LATE: null },
  leidas: { PRE: null, EARLY: null, LATE: null },   // Float32Array de cada fase que se ha leido bien
  ejemploActivo: null,         // indice del ejemplo cargado (o null si se subio a mano)
  faseVisor: "EARLY", animando: null, historial: [], ultimo: null,
};

class EntradaNoValida extends Error {
  constructor(mensaje, comprobaciones) { super(mensaje); this.comprobaciones = comprobaciones || []; }
}

// ------------------------------------------------------------------------------------------------ utilidades
const $ = (id) => document.getElementById(id);
const coma = (x, d = 3) => Number(x).toFixed(d).replace(".", ",");
const pct = (x, d = 0) => `${(100 * x).toFixed(d).replace(".", ",")} %`;
const sigmoide = (z) => 1 / (1 + Math.exp(-z));
const NS = "http://www.w3.org/2000/svg";
function s(etiqueta, atributos = {}, padre = null) {
  const e = document.createElementNS(NS, etiqueta);
  for (const [k, v] of Object.entries(atributos)) {
    if (k === "texto") e.textContent = v; else e.setAttribute(k, v);
  }
  if (padre) padre.appendChild(e);
  return e;
}
const escala = (d0, d1, r0, r1) => (v) => r0 + (v - d0) * (r1 - r0) / (d1 - d0);
const escapar = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const PREVALENCIA = D ? D.validacion.pacientes.reduce((t, q) => t + q[1], 0) / D.validacion.pacientes.length : NaN;
function vaciar(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

// descuadre (tooltip) comun a todas las graficas: cualquier elemento con data-tip
const descuadre = $("descuadre");
document.addEventListener("mousemove", (ev) => {
  const t = ev.target.closest ? ev.target.closest("[data-tip]") : null;
  if (!t) { descuadre.classList.remove("visible"); return; }
  descuadre.innerHTML = t.getAttribute("data-tip");
  const x = Math.min(ev.clientX + 14, window.innerWidth - descuadre.offsetWidth - 8);
  descuadre.style.left = `${x}px`;
  descuadre.style.top = `${ev.clientY + 14}px`;
  descuadre.classList.add("visible");
});

// ------------------------------------------------------------------------------------------------ lectura de PNG
function formatoDe(b) {
  if (b.length >= 8 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47 &&
      b[4] === 0x0d && b[5] === 0x0a && b[6] === 0x1a && b[7] === 0x0a) return "PNG";
  if (b[0] === 0xff && b[1] === 0xd8) return "JPEG";
  if (b[0] === 0x47 && b[1] === 0x49 && b[2] === 0x46) return "GIF";
  if (b[0] === 0x42 && b[1] === 0x4d) return "BMP";
  if (b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46) return "WEBP";
  return null;
}

// Lee UNA fase igual que utils_caso.cargar_fase; devuelve {x, comprobaciones} o lanza EntradaNoValida.
function leerFase(buffer, nombre) {
  const c = [];
  const b = new Uint8Array(buffer);
  if (b.length === 0) throw new EntradaNoValida(`${nombre}: el fichero está vacío.`, [{ t: "vacío", ok: false }]);
  const formato = formatoDe(b);
  c.push({ t: formato === "PNG" ? "PNG" : (formato || "formato desconocido"), ok: formato === "PNG" });
  if (formato !== "PNG") throw new EntradaNoValida(`${nombre}: debe ser un PNG (se ha recibido ${formato || "un formato desconocido"}).`, c);
  const mb = b.length / 2 ** 20;
  c.push({ t: `${coma(mb, mb < 0.1 ? 2 : 1)} MB`, ok: b.length <= TAM_MAX });
  if (b.length > TAM_MAX) throw new EntradaNoValida(`${nombre}: el fichero ocupa demasiado (${coma(mb, 1)} MB; máximo 5 MB).`, c);
  let img, rgba;
  try { img = UPNG.decode(buffer); } catch (e) { img = null; }
  if (!img || !Number.isInteger(img.width) || !Number.isInteger(img.height) || !img.data || !img.data.length) {
    c.push({ t: "dañado", ok: false });
    throw new EntradaNoValida(`${nombre}: no se puede leer como imagen (el fichero está dañado).`, c);
  }
  const medidas = img.width === LADO && img.height === LADO;
  c.push({ t: `${img.width}×${img.height}`, ok: medidas });
  if (!medidas) throw new EntradaNoValida(`${nombre}: debe medir ${LADO}x${LADO} píxeles (mide ${img.width}x${img.height}).`, c);
  if (img.depth > 8) {
    c.push({ t: `${img.depth} bits`, ok: false });
    throw new EntradaNoValida(`${nombre}: el PNG tiene ${img.depth} bits por canal; el modelo se entrenó con PNG de 8 bits.`, c);
  }
  try { rgba = new Uint8Array(UPNG.toRGBA8(img)[0]); } catch (e) {
    c.push({ t: "dañado", ok: false });
    throw new EntradaNoValida(`${nombre}: no se puede leer como imagen (el fichero está dañado).`, c);
  }
  if (rgba.length !== LADO * LADO * 4) throw new EntradaNoValida(`${nombre}: datos de imagen incompletos.`, c);
  const x = new Float32Array(LADO * LADO);
  for (let i = 0, j = 0; i < x.length; i++, j += 4) {
    x[i] = ((rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >> 16) / 255;   // Pillow convert("L")
  }
  for (let i = 0; i < x.length; i++) if (!Number.isFinite(x[i])) throw new EntradaNoValida(`${nombre}: contiene valores no finitos.`, c);
  c.push({ t: "valores finitos", ok: true });
  return { x, comprobaciones: c };
}

function mediaDe(arr, mascara) {
  let s_ = 0, n = 0;
  for (let i = 0; i < arr.length; i++) if (!mascara || mascara[i]) { s_ += arr[i]; n++; }
  return n ? s_ / n : NaN;
}
function desviacion(arr) {
  const m = mediaDe(arr); let s_ = 0;
  for (let i = 0; i < arr.length; i++) s_ += (arr[i] - m) ** 2;
  return Math.sqrt(s_ / arr.length);
}

// Los mismos avisos que predecir.avisos
function avisos(f) {
  const lista = [], tejido = new Uint8Array(f.PRE.length);
  let n = 0;
  for (let i = 0; i < f.PRE.length; i++) if (f.PRE[i] > 0.1) { tejido[i] = 1; n++; }
  if (n < 100) lista.push("La imagen PRE es casi toda negra: ¿es un corte de resonancia de mama?");
  else if (mediaDe(f.EARLY, tejido) <= mediaDe(f.PRE, tejido)) {
    lista.push("En el tejido, EARLY no es más brillante que PRE (en el dataset lo es en el 100 % de las pacientes): ¿están las fases en el orden correcto?");
  }
  if (FASES.some((k) => desviacion(f[k]) < 1e-3)) lista.push("Alguna fase es una imagen prácticamente uniforme.");
  for (const k of FASES) {
    const m = mediaDe(f[k]);
    if (m > 0.8) lista.push(`La fase ${k} es casi toda blanca (brillo medio ${coma(m, 2)}): puede ser una imagen defectuosa y la predicción no es fiable.`);
  }
  return lista;
}

// ------------------------------------------------------------------------------------------------ modelo
async function sha256Hex(partes) {
  const total = partes.reduce((t, p) => t + p.byteLength, 0), junto = new Uint8Array(total);
  let pos = 0;
  for (const p of partes) { junto.set(new Uint8Array(p), pos); pos += p.byteLength; }
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", junto))).map((v) => v.toString(16).padStart(2, "0")).join("");
}

function ponerEstado(texto, clase, progreso) {
  $("estado-texto").textContent = texto;
  $("estado").className = `pastilla ${clase || ""}`;
  if (progreso !== undefined) $("estado-progreso").style.width = `${100 * progreso}%`;
}

async function cargarModelo() {
  if (DEMO) {
    ponerEstado("Maqueta · sin modelos", "", 1);
    estado.listo = true;
    actualizarBotonPredecir();
    return;
  }
  if (!W) { ponerEstado("Faltan los modelos (datos/web.js): ejecuta exportar_web.py", "fallo", 0); $("r-checksum").textContent = "—"; return; }
  try {
    ort.env.wasm.numThreads = 1;     // GitHub Pages no permite hilos de WebAssembly; resultado reproducible
    const buffers = [];
    for (let i = 0; i < W.modelos.length; i++) {
      ponerEstado(`Cargando red ${i + 1} de ${W.modelos.length}`, "", i / W.modelos.length);
      const r = await fetch(`modelos/${W.modelos[i]}`);
      if (!r.ok) throw new Error(`modelos/${W.modelos[i]}: ${r.status}`);
      const buf = await r.arrayBuffer();
      buffers.push(buf);
      estado.sesiones.push(await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] }));
    }
    const m = D.modelo;
    const texto = new TextEncoder().encode(`${m.umbral}|${m.calibracion.a}|${m.calibracion.b}`);
    estado.checksumWeb = (await sha256Hex([...buffers, texto.buffer])).slice(0, 16);
    const ok = estado.checksumWeb === W.checksum_web;
    ponerEstado(ok ? `Modelo listo · ${W.modelos.length} redes · ${estado.checksumWeb}` : `Checksum distinto (${estado.checksumWeb})`, ok ? "listo" : "fallo", 1);
    estado.listo = true;
    $("verificar").disabled = false;
    const marca = `${estado.checksumWeb} <span class="${ok ? "bien" : "mal"}">${ok ? "✓" : "✗"}</span>`;
    $("cs-calc").innerHTML = marca; $("r-checksum").innerHTML = marca;
    actualizarBotonPredecir();
  } catch (e) {
    console.error(e);
    ponerEstado(`No se ha podido cargar el modelo (${e.message})`, "fallo", 0);
    $("r-checksum").textContent = "—";
  }
}

function hash01(f) {   // numero pseudoaleatorio estable a partir de la imagen (solo para la maqueta)
  let h = 2166136261;
  for (const k of FASES) for (let i = 0; i < f[k].length; i += 97) { h ^= Math.round(f[k][i] * 255); h = Math.imul(h, 16777619); }
  return ((h >>> 0) % 100000) / 100000;
}

async function predecirCorte(f) {
  const { umbral, calibracion } = D.modelo;
  const t0 = performance.now();
  let probs;
  if (DEMO) {
    const base = 0.24 + 0.22 * hash01(f);
    probs = Array.from({ length: 15 }, (_, i) => Math.min(0.95, Math.max(0.02, base + 0.05 * Math.sin(7.3 * i + 10 * base))));
    await new Promise((r) => setTimeout(r, 180));
  } else {
    const x = new Float32Array(3 * LADO * LADO);
    FASES.forEach((k, j) => x.set(f[k], j * LADO * LADO));     // canales: PRE, EARLY, LATE
    const tensor = new ort.Tensor("float32", x, [1, 3, LADO, LADO]);
    probs = [];
    for (const sesion of estado.sesiones) probs.push(sigmoide((await sesion.run({ imagen: tensor })).logit.data[0]));
  }
  let m = probs.reduce((a, b) => a + b, 0) / probs.length;
  m = Math.min(Math.max(m, 1e-6), 1 - 1e-6);
  const p = sigmoide(calibracion.a * Math.log(m / (1 - m)) + calibracion.b);
  return { probabilidad: p, clase: p >= umbral ? "pCR" : "no pCR", media: m, probs, segundos: (performance.now() - t0) / 1000 };
}

// ------------------------------------------------------------------------------------------------ imagenes
const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function colorInferno(t) {
  const x = Math.min(Math.max(t, 0), 1) * (INFERNO.length - 1);
  const i = Math.min(Math.floor(x), INFERNO.length - 2), r = x - i;
  return INFERNO[i].map((c, k) => Math.round(c + r * (INFERNO[i + 1][k] - c)));
}
function pintarGris(ctx, datos) {
  const img = ctx.createImageData(LADO, LADO);
  for (let i = 0, j = 0; i < datos.length; i++, j += 4) {
    const v = Math.round(Math.min(Math.max(datos[i], 0), 1) * 255);
    img.data[j] = img.data[j + 1] = img.data[j + 2] = v; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function pintarRealce(ctx, pre, early) {
  const img = ctx.createImageData(LADO, LADO), r = new Float32Array(pre.length);
  let max = 1e-3;
  for (let i = 0; i < pre.length; i++) { r[i] = Math.max(early[i] - pre[i], 0); if (r[i] > max) max = r[i]; }
  for (let i = 0, j = 0; i < r.length; i++, j += 4) {
    const [a, b, c] = colorInferno(r[i] / max);
    img.data[j] = a; img.data[j + 1] = b; img.data[j + 2] = c; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function pintarRecorte(ctx) {
  if (!$("ver-recorte").checked) return;
  const ini = (LADO - RECORTE) / 2;
  ctx.save(); ctx.strokeStyle = "#ffd60a"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
  ctx.strokeRect(ini + 0.75, ini + 0.75, RECORTE - 1.5, RECORTE - 1.5); ctx.restore();
}
function mezcla(a, b, t) { const r = new Float32Array(a.length); for (let i = 0; i < a.length; i++) r[i] = a[i] + t * (b[i] - a[i]); return r; }

function dibujarVisor(datos, etiqueta) {
  const ctx = $("visor").getContext("2d");
  if (etiqueta === "REALCE") pintarRealce(ctx, estado.fases.PRE, estado.fases.EARLY);
  else pintarGris(ctx, datos || estado.fases[etiqueta]);
  pintarRecorte(ctx);
  $("visor-etiqueta").hidden = false;
  $("visor-etiqueta").textContent = etiqueta === "REALCE" ? "Realce (EARLY − PRE)" : etiqueta;
}
const DESCRIPCION_FASE = { PRE: "antes del contraste", EARLY: "contraste temprano", LATE: "contraste tardío", REALCE: "cuánto brilla más en EARLY que en PRE" };
function mostrarFase(fase) {
  estado.faseVisor = fase;
  $("visor-info").textContent = estado.fases ? `${fase === "REALCE" ? "Realce" : fase}: ${DESCRIPCION_FASE[fase]}` : "";
  document.querySelectorAll("[data-fase-visor]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.faseVisor === fase)));
  if (estado.fases) dibujarVisor(null, fase);
}
function animarSecuencia() {
  if (!estado.fases) return;
  if (estado.animando) { cancelAnimationFrame(estado.animando); estado.animando = null; mostrarFase(estado.faseVisor); return; }
  const tramos = [["PRE", "EARLY"], ["EARLY", "LATE"]], dur = 1300, pausa = 450, inicio = performance.now();
  const total = 2 * (tramos.length * (dur + pausa));
  const paso = (ahora) => {
    const t = (ahora - inicio) % (tramos.length * (dur + pausa));
    const k = Math.floor(t / (dur + pausa)), dentro = t - k * (dur + pausa);
    const [a, b] = tramos[k];
    const u = Math.min(dentro / dur, 1), suave = u * u * (3 - 2 * u);
    dibujarVisor(mezcla(estado.fases[a], estado.fases[b], suave), suave < 0.5 ? a : b);
    $("visor-etiqueta").textContent = suave < 0.02 ? a : suave > 0.98 ? b : `${a} → ${b}`;
    if (ahora - inicio < total) estado.animando = requestAnimationFrame(paso);
    else { estado.animando = null; mostrarFase("LATE"); }
  };
  estado.animando = requestAnimationFrame(paso);
}
function miniatura(destino, datos) {
  const c = document.createElement("canvas"); c.width = c.height = LADO;
  pintarGris(c.getContext("2d"), datos);
  destino.innerHTML = ""; destino.appendChild(c);
  return c;
}

// ------------------------------------------------------------------------------------------------ graficas
function ejes(svg, { x, y, xt, yt, fx = (v) => coma(v, 2), fy = (v) => coma(v, 2), xl, yl, marco }) {
  const [l, r, t, b] = marco;
  const gr = s("g", { class: "rejilla" }, svg), ge = s("g", { class: "eje" }, svg);
  for (const v of yt) { s("line", { x1: l, x2: r, y1: y(v), y2: y(v) }, gr); s("text", { x: l - 6, y: y(v) + 4, "text-anchor": "end", texto: fy(v) }, ge); }
  for (const v of xt) s("text", { x: x(v), y: b + 15, "text-anchor": "middle", texto: fx(v) }, ge);
  s("line", { x1: l, x2: r, y1: b, y2: b }, ge);
  if (xl) s("text", { class: "titulo-eje", x: (l + r) / 2, y: b + 30, "text-anchor": "middle", texto: xl }, svg);
  if (yl) s("text", { class: "titulo-eje", x: 12, y: (t + b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(t + b) / 2})`, texto: yl }, svg);
}

function histograma(svg, ancho, alto, umbral, marca) {
  vaciar(svg);
  const pac = D.validacion.pacientes, w = 0.02, x0 = 0.06, x1 = 0.56, nb = Math.round((x1 - x0) / w);
  const bins = Array.from({ length: nb }, () => [0, 0]);
  for (const [p, y] of pac) { const k = Math.min(nb - 1, Math.max(0, Math.floor((p - x0) / w))); bins[k][y]++; }
  const ymax = Math.max(...bins.map(([a, b]) => a + b));
  const marco = [40, ancho - 10, 10, alto - 38];
  const x = escala(x0, x1, marco[0], marco[1]), y = escala(0, ymax * 1.08, marco[3], marco[2]);
  const yt = [0, Math.round(ymax / 2), ymax];
  ejes(svg, { x, y, xt: [0.1, 0.2, 0.3, 0.4, 0.5], yt, fy: (v) => String(v), xl: "probabilidad de pCR (calibrada)", marco });
  bins.forEach(([n0, n1], k) => {
    const a = x0 + k * w, xa = x(a) + 1, wa = x(a + w) - x(a) - 2;
    const tip = `<b>${coma(a, 2)}–${coma(a + w, 2)}</b><br>${n0 + n1} pacientes · ${n1} con pCR${n0 + n1 ? ` (${pct(n1 / (n0 + n1))})` : ""}`;
    if (n0) s("rect", { x: xa, width: wa, y: y(n0), height: y(0) - y(n0), rx: 2, style: "fill:var(--nopcr)", "data-tip": tip }, svg);
    if (n1) s("rect", { x: xa, width: wa, y: y(n0 + n1), height: Math.max(1, y(n0) - y(n0 + n1) - (n0 ? 1.5 : 0)), rx: 2, style: "fill:var(--pcr)", "data-tip": tip }, svg);
  });
  if (umbral !== undefined) {
    s("line", { x1: x(umbral), x2: x(umbral), y1: marco[2], y2: marco[3], style: "stroke:var(--texto-2);stroke-dasharray:4 3;stroke-width:1.5" }, svg);
  }
  if (marca !== undefined) {
    const xm = x(Math.min(Math.max(marca, x0), x1));
    s("line", { x1: xm, x2: xm, y1: marco[2] - 2, y2: marco[3], style: "stroke:var(--texto);stroke-width:2.5" }, svg);
    s("circle", { cx: xm, cy: marco[2] + 2, r: 5, style: "fill:var(--texto)" }, svg);
    s("text", { class: "marca-texto", x: xm + (xm > ancho * 0.7 ? -8 : 8), y: marco[2] + 6, "text-anchor": xm > ancho * 0.7 ? "end" : "start", texto: "esta muestra" }, svg);
  }
}

function medidor(p, umbral) {
  const svg = $("g-medidor"); vaciar(svg);
  const cx = 160, cy = 160, r = 128, ang = (v) => Math.PI * (1 - v);
  const punto = (v, rr = r) => [cx + rr * Math.cos(ang(v)), cy - rr * Math.sin(ang(v))];
  const arco = (a, b, rr = r) => { const [x1, y1] = punto(a, rr), [x2, y2] = punto(b, rr); return `M${x1} ${y1} A${rr} ${rr} 0 0 1 ${x2} ${y2}`; };
  s("path", { d: arco(0, 1), style: "fill:none;stroke:var(--borde);stroke-width:10" }, svg);
  s("path", { d: arco(0, umbral), style: "fill:none;stroke:var(--nopcr);stroke-width:10" }, svg);
  s("path", { d: arco(umbral, 1), style: "fill:none;stroke:var(--pcr);stroke-width:10" }, svg);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    const [tx, ty] = punto(v, r - 26);
    s("text", { x: tx, y: ty + 4, "text-anchor": "middle", texto: coma(v, v % 0.5 ? 2 : 1).replace(",0", "") }, svg);
  }
  const [u1x, u1y] = punto(umbral, r + 14), [u2x, u2y] = punto(umbral, r - 14);
  s("line", { x1: u1x, y1: u1y, x2: u2x, y2: u2y, style: "stroke:var(--texto);stroke-width:2" }, svg);
  const [ux, uy] = punto(umbral, r + 18), izquierda = umbral < 0.5;
  s("text", { class: "marca-texto", x: ux + (izquierda ? -6 : 6), y: uy, "text-anchor": izquierda ? "end" : "start",
              texto: `umbral ${coma(umbral, 3)}` }, svg);
  const g = s("g", { class: "aguja", transform: `rotate(-90 ${cx} ${cy})` }, svg);
  s("path", { d: `M${cx - 3} ${cy} L${cx} ${cy - r + 20} L${cx + 3} ${cy} Z`, style: "fill:var(--texto)" }, g);
  s("circle", { cx, cy, r: 6, style: "fill:var(--panel);stroke:var(--texto);stroke-width:2" }, svg);
  animar(1000, (k) => g.setAttribute("transform", `rotate(${-90 + 180 * p * k} ${cx} ${cy})`));
}

// Anima de 0 a 1 en `ms` con una curva suave. Usa el reloj normal y un temporizador deja siempre el valor final
// (aunque la pestana este en segundo plano y no pinte fotogramas).
function animar(ms, pintar) {
  if (SIN_MOVIMIENTO) { pintar(1); return; }
  const ini = performance.now();
  let hecho = false;
  const fin = () => { if (!hecho) { hecho = true; pintar(1); } };
  const paso = () => {
    if (hecho) return;
    const u = (performance.now() - ini) / ms;
    if (u >= 1) { fin(); return; }
    pintar(1 - (1 - Math.max(u, 0)) ** 3);
    requestAnimationFrame(paso);
  };
  pintar(0);
  requestAnimationFrame(paso);
  setTimeout(fin, ms + 80);
}

let ultimaCuenta = 0;
function contarHasta(elemento, valor, d = 3, ms = 900) {
  const id = ++ultimaCuenta;   // si llega otra prediccion antes de acabar, la animacion anterior deja de escribir
  animar(ms, (k) => { if (id === ultimaCuenta) elemento.textContent = coma(valor * k, d); });
}

function graficaRedes(probs) {
  const svg = $("g-redes"); vaciar(svg);
  const marco = [52, 350, 4, 124], xmax = Math.max(0.6, ...probs) + 0.02, xmin = Math.max(0, Math.min(...probs) - 0.08);
  const x = escala(xmin, xmax, marco[0], marco[1]), fila = (k) => marco[2] + 12 + k * 20;
  const ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].filter((v) => v >= xmin && v <= xmax);
  const gr = s("g", { class: "rejilla" }, svg), ge = s("g", { class: "eje" }, svg);
  for (const v of ticks) { s("line", { x1: x(v), x2: x(v), y1: marco[2], y2: marco[3] }, gr); s("text", { x: x(v), y: marco[3] + 16, "text-anchor": "middle", texto: coma(v, 1) }, ge); }
  for (let k = 0; k < 5; k++) s("text", { x: marco[0] - 8, y: fila(k) + 4, "text-anchor": "end", texto: `fold ${k}` }, ge);
  const media = probs.reduce((a, b) => a + b, 0) / probs.length;
  s("line", { x1: x(media), x2: x(media), y1: marco[2], y2: marco[3], style: "stroke:var(--acento);stroke-width:2" }, svg);
  s("text", { class: "marca-texto", x: x(media) + 5, y: marco[3] - 2, texto: `media ${coma(media, 3)}` }, svg);
  probs.forEach((p, i) => {
    const sem = Math.floor(i / 5), fold = i % 5;
    s("circle", { cx: x(p), cy: fila(fold) + (sem - 1) * 5, r: 4.5, style:"fill:var(--texto-2);stroke:var(--panel);stroke-width:2",
                  "data-tip": `<b>fold ${fold} · semilla ${SEMILLAS[sem]}</b><br>probabilidad ${coma(p, 3)}` }, svg);
  });
}

function graficaCinetica(f) {
  const svg = $("g-cinetica"); vaciar(svg);
  const ini = (LADO - RECORTE) / 2, centro = new Uint8Array(LADO * LADO), tejido = new Uint8Array(LADO * LADO);
  for (let fy = 0; fy < LADO; fy++) for (let fx = 0; fx < LADO; fx++) {
    const i = fy * LADO + fx;
    if (fx >= ini && fx < ini + RECORTE && fy >= ini && fy < ini + RECORTE) centro[i] = 1;
    if (f.PRE[i] > 0.1) tejido[i] = 1;
  }
  const series = [
    { nombre: "zona central (la que mira la red)", color: "var(--serie-3)", v: FASES.map((k) => mediaDe(f[k], centro)) },
    { nombre: "todo el tejido", color: "var(--texto-3)", v: FASES.map((k) => mediaDe(f[k], tejido)) },
  ];
  const ymax = Math.max(...series.flatMap((q) => q.v)) * 1.18 || 1;
  const marco = [46, 330, 14, 160];
  const x = (k) => marco[0] + 30 + k * ((marco[1] - marco[0] - 60) / 2), y = escala(0, ymax, marco[3], marco[2]);
  const gr = s("g", { class: "rejilla" }, svg), ge = s("g", { class: "eje" }, svg);
  for (const v of [0, ymax / 2, ymax]) { s("line", { x1: marco[0], x2: marco[1], y1: y(v), y2: y(v) }, gr); s("text", { x: marco[0] - 6, y: y(v) + 4, "text-anchor": "end", texto: coma(v, 2) }, ge); }
  FASES.forEach((k, j) => s("text", { x: x(j), y: marco[3] + 16, "text-anchor": "middle", texto: k }, ge));
  s("text", { class: "titulo-eje", x: 12, y: 87, "text-anchor": "middle", transform: "rotate(-90 12 87)", texto: "brillo medio" }, svg);
  for (const q of series) {
    s("polyline", { points: q.v.map((v, j) => `${x(j)},${y(v)}`).join(" "), style: `fill:none;stroke:${q.color};stroke-width:2.5` }, svg);
    q.v.forEach((v, j) => s("circle", { cx: x(j), cy: y(v), r: 5, style: `fill:${q.color};stroke:var(--panel);stroke-width:2`,
                                       "data-tip": `<b>${FASES[j]}</b> · ${q.nombre}<br>brillo medio ${coma(v, 3)}` }, svg));
  }
  const [pre, early, late] = series[0].v;
  const realce = (early - pre) / Math.max(pre, 1e-6), cambio = (late - early) / Math.max(early, 1e-6);
  const forma = cambio > 0.05 ? "sigue subiendo" : cambio < -0.05 ? "lavado (baja tras el pico)" : "meseta";
  const g = s("g", {}, svg);
  s("text", { class: "marca-texto", x: 350, y: 40, texto: "Zona central" }, g);
  const conSigno = (v) => { const r = Math.round(100 * v); return `${r > 0 ? "+" : ""}${r === 0 ? 0 : r} %`; };
  s("text", { x: 350, y: 62, texto: `realce temprano: ${conSigno(realce)}` }, g);
  s("text", { x: 350, y: 82, texto: `de EARLY a LATE: ${conSigno(cambio)}` }, g);
  s("text", { x: 350, y: 102, texto: `forma: ${forma}` }, g);
  s("text", { x: 350, y: 130, texto: "Descriptivo, no es una" }, g);
  s("text", { x: 350, y: 146, texto: "valoración clínica." }, g);
  $("l-cinetica").innerHTML = series.map((q) => `<span><i class="linea" style="--c:${q.color}"></i>${q.nombre}</span>`).join("");
}

// ------------------------------------------------------------------------------------------------ vista Predecir
function actualizarBotonPredecir() {
  $("predecir").disabled = !(estado.listo && estado.fases);
}

function ponerChips(fase, comprobaciones) {
  document.querySelector(`[data-chips="${fase}"]`).innerHTML =
    comprobaciones.map((c) => `<span class="chip ${c.ok ? "bien" : "mal"}">${c.ok ? "✓" : "✗"} ${escapar(c.t)}</span>`).join("");
}

function cineticaVacia() {
  const svg = $("g-cinetica"); vaciar(svg);
  const ge = s("g", { class: "eje" }, svg), x = (k) => 76 + k * 112;
  s("line", { x1: 46, x2: 330, y1: 160, y2: 160 }, ge);
  FASES.forEach((k, j) => s("text", { x: x(j), y: 176, "text-anchor": "middle", texto: k }, ge));
  s("polyline", { points: `${x(0)},140 ${x(1)},70 ${x(2)},84`, style: "fill:none;stroke:var(--borde-fuerte);stroke-width:2;stroke-dasharray:5 5" }, svg);
  s("text", { x: 350, y: 80, texto: "Al cargar una muestra: cómo" }, svg);
  s("text", { x: 350, y: 98, texto: "cambia el brillo de la zona" }, svg);
  s("text", { x: 350, y: 116, texto: "central de PRE a EARLY y LATE." }, svg);
  $("l-cinetica").innerHTML = "";
}

// La muestra esta lista cuando las tres fases se han leido bien: entonces se ve en el visor y en la curva de realce
function prepararMuestra() {
  const ok = FASES.every((k) => estado.leidas[k]);
  estado.fases = ok ? { PRE: estado.leidas.PRE, EARLY: estado.leidas.EARLY, LATE: estado.leidas.LATE } : null;
  if (estado.animando) { cancelAnimationFrame(estado.animando); estado.animando = null; }
  $("visor-vacio").hidden = ok; $("visor-etiqueta").hidden = !ok; $("play").disabled = !ok;
  if (ok) { mostrarFase(estado.faseVisor); graficaCinetica(estado.fases); }
  else { $("visor").getContext("2d").clearRect(0, 0, LADO, LADO); $("visor-info").textContent = ""; cineticaVacia(); }
  actualizarBotonPredecir();
}

function ponerFichero(fase, buffer, nombre) {
  estado.buffers[fase] = buffer; estado.nombres[fase] = nombre; estado.leidas[fase] = null; estado.ejemploActivo = null;
  const ranura = document.querySelector(`.ranura[data-fase="${fase}"]`), et = document.querySelector(`[data-estado="${fase}"]`);
  const mini = document.querySelector(`[data-mini="${fase}"]`);
  ranura.classList.remove("cargada", "mal");
  if (!buffer) { et.textContent = "Arrastra o elige un PNG"; mini.textContent = fase; ponerChips(fase, []); prepararMuestra(); return; }
  try {
    const { x, comprobaciones } = leerFase(buffer, fase);
    estado.leidas[fase] = x;
    ranura.classList.add("cargada"); et.textContent = nombre; ponerChips(fase, comprobaciones); miniatura(mini, x);
  } catch (e) {
    ranura.classList.add("mal");
    et.textContent = e instanceof EntradaNoValida ? e.message : `${fase}: error inesperado al leer la imagen.`;
    ponerChips(fase, e.comprobaciones || []); mini.textContent = "✗";
  }
  prepararMuestra();
}

function cargarFasesDirectas(fases, nombre, indice) {   // ejemplos de la maqueta (sin PNG)
  for (const k of FASES) {
    estado.buffers[k] = null; estado.nombres[k] = `${nombre} · ${k}`; estado.leidas[k] = fases[k];
    const ranura = document.querySelector(`.ranura[data-fase="${k}"]`);
    ranura.classList.remove("mal"); ranura.classList.add("cargada");
    document.querySelector(`[data-estado="${k}"]`).textContent = `${nombre} (${k})`;
    ponerChips(k, [{ t: "fantasma sintético", ok: true }, { t: "256×256", ok: true }]);
    miniatura(document.querySelector(`[data-mini="${k}"]`), fases[k]);
  }
  document.querySelectorAll('.ranura input[type="file"]').forEach((i) => { i.value = ""; });
  prepararMuestra();
  estado.ejemploActivo = indice;
}

function mostrarErrores(lista) {
  const caja = $("errores");
  caja.hidden = !lista.length;
  caja.innerHTML = lista.length ? `<ul>${lista.map((e) => `<li>${escapar(e)}</li>`).join("")}</ul>` : "";
}

async function predecir() {
  mostrarErrores([]);
  const fases = estado.fases;
  if (!fases) return;
  const boton = $("predecir");
  boton.classList.add("calculando"); boton.disabled = true;
  try {
    const r = await predecirCorte(fases);
    mostrarResultado(r, avisos(fases));
    anadirHistorial(r, fases);
  } catch (e) {
    console.error(e); mostrarErrores([`No se ha podido calcular la predicción (${e.message}).`]);
  } finally {
    boton.classList.remove("calculando"); actualizarBotonPredecir();
  }
}

// Antes de la primera prediccion: el contexto (sin marca) y una nota en la tarjeta de las 15 redes
function resultadoInicial() {
  $("resultado").hidden = true; $("resultado-vacio").hidden = false; $("r-hora").textContent = "";
  histograma($("g-contexto"), 360, 170, D.modelo.umbral);
  $("r-contexto").innerHTML = `Así se reparten las probabilidades de las pacientes de validación (el ${pct(PREVALENCIA)} tuvo pCR). ` +
                              "Al predecir, una línea marcará dónde cae la muestra.";
  const svg = $("g-redes"); vaciar(svg);
  const ge = s("g", { class: "eje" }, svg);
  for (let k = 0; k < 5; k++) s("text", { x: 44, y: 20 + k * 20, "text-anchor": "end", texto: `fold ${k}` }, ge);
  s("text", { x: 200, y: 62, "text-anchor": "middle", texto: "Al predecir: la probabilidad que da" }, svg);
  s("text", { x: 200, y: 80, "text-anchor": "middle", texto: "cada una de las 15 redes" }, svg);
  $("r-tiempo").textContent = "—";
}

function mostrarResultado(r, listaAvisos) {
  const u = D.modelo.umbral;
  $("resultado-vacio").hidden = true; $("resultado").hidden = false;
  medidor(r.probabilidad, u);
  contarHasta($("r-prob"), r.probabilidad);
  const pcr = r.clase === "pCR";
  $("r-clase").className = pcr ? "pcr" : "nopcr";
  $("r-clase").textContent = pcr ? "pCR = 1 · respuesta completa probable" : "pCR = 0 · respuesta completa improbable";
  $("r-frase").textContent = `La probabilidad calibrada (${coma(r.probabilidad, 3)}) está ${pcr ? "por encima" : "por debajo"} del umbral ${coma(u, 3)}.`;
  $("r-avisos").innerHTML = listaAvisos.map((a) => `<li>${escapar(a)}</li>`).join("");
  $("r-hora").textContent = new Date().toLocaleTimeString("es-ES");
  histograma($("g-contexto"), 360, 170, u, r.probabilidad);
  const pac = D.validacion.pacientes, cerca = pac.filter(([p]) => Math.abs(p - r.probabilidad) <= 0.02);
  const menores = pac.filter(([p]) => p < r.probabilidad).length;
  $("r-contexto").innerHTML = (cerca.length >= 10
    ? `Entre las <b>${cerca.length}</b> pacientes de validación con una probabilidad parecida (±0,02), el <b>${pct(cerca.filter(([, y]) => y).length / cerca.length)}</b> tuvo pCR (en total, el ${pct(PREVALENCIA)}). `
    : "Hay pocas pacientes de validación con una probabilidad tan extrema. ") + `Su probabilidad supera a la del ${pct(menores / pac.length)} de ellas.`;
  graficaRedes(r.probs);
  $("r-tiempo").textContent = DEMO ? "— (maqueta)" : `${coma(r.segundos * 1000, 0)} ms · 15 redes`;
}

function anadirHistorial(r, fases) {
  const n = estado.historial.length + 1, hora = new Date().toLocaleTimeString("es-ES");
  const nombre = estado.ejemploActivo !== null && !DEMO ? W.ejemplos[estado.ejemploActivo].sample_id
    : (estado.nombres.EARLY || "muestra").replace(/_EARLY\.png$/i, "").replace(/ · EARLY$/, "");
  const fila = { n, hora, nombre, probabilidad: r.probabilidad, clase: r.clase, ms: r.segundos * 1000, checksum: DEMO ? "maqueta" : estado.checksumWeb };
  estado.historial.push(fila);
  const tbody = $("historial");
  if (n === 1) tbody.innerHTML = "";
  const tr = document.createElement("tr"); tr.className = "fila-nueva";
  const td = (texto, clase) => { const c = document.createElement("td"); if (clase) c.className = clase; c.innerHTML = texto; return c; };
  tr.append(td(String(n)), td(hora));
  const celda = td(""); const c = document.createElement("canvas"); c.width = c.height = LADO; pintarGris(c.getContext("2d"), fases.EARLY); celda.appendChild(c); tr.append(celda);
  tr.append(td(escapar(nombre)), td(coma(r.probabilidad, 3), "num"),
            td(`<span class="punto-clase" style="--c:var(${r.clase === "pCR" ? "--pcr" : "--nopcr"})">${r.clase === "pCR" ? "pCR = 1" : "pCR = 0"}</span>`),
            td(DEMO ? "—" : `${coma(fila.ms, 0)} ms`, "num"), td(fila.checksum, "num"));
  tbody.prepend(tr);
  $("descargar").disabled = false; $("vaciar").disabled = false;
}

function descargarHistorial() {
  const filas = [["n", "hora", "muestra", "probabilidad", "clase", "umbral", "tiempo_ms", "checksum"]];
  for (const h of estado.historial) filas.push([h.n, h.hora, h.nombre, coma(h.probabilidad, 6), h.clase === "pCR" ? "pCR=1" : "pCR=0", coma(D.modelo.umbral, 6), DEMO ? "" : coma(h.ms, 0), h.checksum]);
  const csv = filas.map((f) => f.join(";")).join("\r\n");
  const bom = String.fromCharCode(0xfeff);   // para que Excel lea bien las tildes
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([bom + csv], { type: "text/csv;charset=utf-8" }));
  a.download = "historial_pcr.csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function limpiar() {
  for (const k of FASES) ponerFichero(k, null, null);
  document.querySelectorAll('.ranura input[type="file"]').forEach((i) => { i.value = ""; });
  mostrarErrores([]);
  resultadoInicial();
}

// ---- ejemplos ----
function aleatorio(semilla) { let a = semilla >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function fantasma(semilla) {   // corte sintetico (solo maqueta): mama, tejido fibroglandular, tumor que realza y corazon
  const rnd = aleatorio(semilla), f = { PRE: new Float32Array(LADO * LADO), EARLY: new Float32Array(LADO * LADO), LATE: new Float32Array(LADO * LADO) };
  const tx = 128 + (rnd() - 0.5) * 50, ty = 132 + (rnd() - 0.5) * 40, tr = 14 + rnd() * 14, lavado = rnd() < 0.5;
  const fase1 = rnd() * 6, fase2 = rnd() * 6;
  for (let y = 0; y < LADO; y++) for (let x = 0; x < LADO; x++) {
    const i = y * LADO + x, ruido = () => (rnd() - 0.5) * 0.04;
    const dentro = ((x - 128) / 118) ** 2 + ((y - 70) / 175) ** 2 < 1 && y > 52;
    const corazon = ((x - 128) / 70) ** 2 + ((y - 22) / 26) ** 2 < 1;
    let pre = 0.02, early = 0.02, late = 0.02;
    if (dentro) {
      const textura = 0.5 + 0.5 * Math.sin(x / 9 + fase1) * Math.cos(y / 11 + fase2);
      pre = 0.18 + 0.12 * textura; early = pre + 0.05 + 0.04 * textura; late = early + 0.01;
      const d = Math.hypot(x - tx, y - ty) / tr;
      if (d < 1.15) { const borde = Math.max(0, 1 - Math.max(0, d - 0.85) / 0.3); pre += 0.04 * borde; early += 0.5 * borde; late += (lavado ? 0.36 : 0.56) * borde; }
    }
    if (corazon) { pre = Math.max(pre, 0.22); early = Math.max(early, 0.62); late = Math.max(late, 0.5); }
    f.PRE[i] = Math.min(1, Math.max(0, pre + ruido())); f.EARLY[i] = Math.min(1, Math.max(0, early + ruido())); f.LATE[i] = Math.min(1, Math.max(0, late + ruido()));
  }
  return f;
}

async function bajar(ruta) { const r = await fetch(ruta); if (!r.ok) throw new Error(`${ruta}: ${r.status}`); return r.arrayBuffer(); }

function montarGaleria() {
  const g = $("galeria"); g.innerHTML = "";
  if (DEMO) {
    $("ejemplos-nota").textContent = "fantasmas sintéticos";
    for (let i = 0; i < 6; i++) {
      const f = fantasma(1000 + i * 77), b = document.createElement("button");
      b.className = "ejemplo"; b.type = "button";
      const c = document.createElement("canvas"); c.width = c.height = LADO; pintarGris(c.getContext("2d"), f.EARLY);
      b.append(c); b.insertAdjacentHTML("beforeend", `<span><b>Fantasma ${i + 1}</b><i>sintético</i></span>`);
      b.addEventListener("click", () => cargarFasesDirectas(f, `Fantasma ${i + 1}`, i));
      g.append(b);
    }
    return;
  }
  if (!W) { g.innerHTML = '<p class="texto-tenue" style="grid-column:1/-1;margin:0">Los ejemplos se generan con exportar_web.py.</p>'; return; }
  $("ejemplos-nota").textContent = "pacientes de entrenamiento";
  W.ejemplos.forEach((e, i) => {
    const b = document.createElement("button"); b.className = "ejemplo"; b.type = "button";
    b.innerHTML = `<img src="${e.rutas.EARLY}" alt="" loading="lazy"><span><b>pCR real = ${e.pCR}</b><i>${e.patient_id}</i></span>`;
    b.title = `${e.sample_id} (paciente de entrenamiento: la mayoría de las redes la vieron)`;
    b.addEventListener("click", async () => {
      mostrarErrores([]);
      try {
        const buffers = await Promise.all(FASES.map((k) => bajar(e.rutas[k])));
        document.querySelectorAll('.ranura input[type="file"]').forEach((inp) => { inp.value = ""; });
        FASES.forEach((k, j) => ponerFichero(k, buffers[j], e.rutas[k].split("/").pop()));
        estado.ejemploActivo = i;
      } catch (err) { mostrarErrores([`No se ha podido cargar el ejemplo (${err.message}).`]); }
    });
    g.append(b);
  });
}

// ------------------------------------------------------------------------------------------------ vista Modelo
function aucMannWhitney(pares) {
  const orden = pares.map(([p, y]) => [p, y]).sort((a, b) => a[0] - b[0]);
  let rango = 0, i = 0, sumaPos = 0, nPos = 0;
  while (i < orden.length) {
    let j = i; while (j + 1 < orden.length && orden[j + 1][0] === orden[i][0]) j++;
    const medio = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) if (orden[k][1] === 1) { sumaPos += medio; nPos++; }
    i = j + 1;
  }
  const nNeg = orden.length - nPos;
  return (sumaPos - nPos * (nPos + 1) / 2) / (nPos * nNeg);
}

function metricasUmbral(u) {
  // las probabilidades de validacion estan guardadas con 6 decimales: se compara a esa precision
  // (el umbral elegido coincide con la probabilidad de una paciente)
  u = Math.round(u * 1e6) / 1e6;
  let vp = 0, fn = 0, vn = 0, fp = 0;
  for (const [p, y] of D.validacion.pacientes) { if (y) { p >= u ? vp++ : fn++; } else { p >= u ? fp++ : vn++; } }
  const div = (a, b) => (b ? a / b : 0), sens = div(vp, vp + fn), esp = div(vn, vn + fp), prec = div(vp, vp + fp), vpn = div(vn, vn + fn);
  const f1p = div(2 * prec * sens, prec + sens), f1n = div(2 * vpn * esp, vpn + esp);
  return { vp, fn, vn, fp, sens, esp, prec, vpn, f1eq: (f1p + f1n) / 2, acc: div(vp + vn, vp + fn + vn + fp) };
}

function curvaRoc() {
  const pares = D.validacion.pacientes, umbrales = [...new Set(pares.map(([p]) => p))].sort((a, b) => b - a);
  const nPos = pares.filter(([, y]) => y).length, nNeg = pares.length - nPos, puntos = [[0, 0]];
  for (const u of umbrales) {
    let vp = 0, fp = 0;
    for (const [p, y] of pares) if (p >= u) { if (y) vp++; else fp++; }
    puntos.push([fp / nNeg, vp / nPos]);
  }
  return puntos;
}
let ROC = null;

function dibujarRoc(u) {
  const svg = $("g-roc"); vaciar(svg);
  const marco = [40, 290, 10, 256], x = escala(0, 1, marco[0], marco[1]), y = escala(0, 1, marco[3], marco[2]);
  ejes(svg, { x, y, xt: [0, 0.5, 1], yt: [0, 0.5, 1], fx: (v) => coma(v, 1), fy: (v) => coma(v, 1), xl: "1 − especificidad", yl: "sensibilidad", marco });
  s("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), style: "stroke:var(--texto-3);stroke-dasharray:4 3" }, svg);
  s("polyline", { points: ROC.map(([a, b]) => `${x(a)},${y(b)}`).join(" "), style: "fill:none;stroke:var(--acento);stroke-width:2" }, svg);
  const m = metricasUmbral(u);
  s("circle", { cx: x(1 - m.esp), cy: y(m.sens), r: 7, style: "fill:var(--acento);stroke:var(--panel);stroke-width:3",
                "data-tip": `umbral ${coma(u, 3)}<br>sensibilidad ${pct(m.sens)} · especificidad ${pct(m.esp)}` }, svg);
  s("text", { class: "marca-texto", x: x(0.55), y: y(0.12), texto: `AUC ${coma(D.modelo.metricas_validacion.auc, 3)}` }, svg);
}

function actualizarUmbral(u) {
  $("u-valor").textContent = coma(u, 3);
  const m = metricasUmbral(u);
  $("u-matriz").innerHTML = `<div></div><div class="cab">predicción<br>pCR = 0</div><div class="cab">predicción<br>pCR = 1</div>
    <div class="cab">real<br>pCR = 0</div><div class="celda acierto"><b>${m.vn}</b>verdaderos negativos</div><div class="celda"><b>${m.fp}</b>falsos positivos</div>
    <div class="cab">real<br>pCR = 1</div><div class="celda"><b>${m.fn}</b>falsos negativos</div><div class="celda acierto"><b>${m.vp}</b>verdaderos positivos</div>`;
  const tarjeta = (k, v) => `<div><span>${k}</span><b>${v}</b></div>`;
  $("u-metricas").innerHTML = tarjeta("sensibilidad", pct(m.sens)) + tarjeta("especificidad", pct(m.esp)) + tarjeta("precisión", pct(m.prec)) +
                              tarjeta("VPN", pct(m.vpn)) + tarjeta("F1 equilibrada", coma(m.f1eq, 2)) + tarjeta("accuracy", pct(m.acc));
  dibujarRoc(u);
  histograma($("g-distribucion"), 440, 190, u);
}

function dibujarCalibracion() {
  const svg = $("g-calibracion"); vaciar(svg);
  const orden = [...D.validacion.pacientes].sort((a, b) => a[0] - b[0]), g = 10, tam = Math.ceil(orden.length / g), grupos = [];
  for (let k = 0; k < g; k++) { const t = orden.slice(k * tam, (k + 1) * tam); if (t.length) grupos.push([mediaDe(t.map((q) => q[0])), mediaDe(t.map((q) => q[1])), t.length]); }
  const lim = Math.max(...grupos.flatMap(([a, b]) => [a, b])) * 1.15;
  const marco = [44, 430, 10, 172], x = escala(0, lim, marco[0], marco[1]), y = escala(0, lim, marco[3], marco[2]);
  const t = [0, 0.1, 0.2, 0.3, 0.4, 0.5].filter((v) => v <= lim);
  ejes(svg, { x, y, xt: t, yt: t, fx: (v) => coma(v, 1), fy: (v) => coma(v, 1), xl: "probabilidad media predicha", yl: "pCR real", marco });
  s("line", { x1: x(0), y1: y(0), x2: x(lim), y2: y(lim), style: "stroke:var(--texto-3);stroke-dasharray:4 3" }, svg);
  s("polyline", { points: grupos.map(([a, b]) => `${x(a)},${y(b)}`).join(" "), style: "fill:none;stroke:var(--acento);stroke-width:2" }, svg);
  grupos.forEach(([a, b, n], k) => s("circle", { cx: x(a), cy: y(b), r: 5.5, style: "fill:var(--acento);stroke:var(--panel);stroke-width:2",
    "data-tip": `<b>grupo ${k + 1}</b> (${n} pacientes)<br>predicho ${coma(a, 3)} · real ${coma(b, 3)}` }, svg));
  const brier = mediaDe(D.validacion.pacientes.map(([p, y]) => (p - y) ** 2));
  s("text", { class: "marca-texto", x: x(lim * 0.04), y: y(lim * 0.92), texto: `Brier ${coma(brier, 4)} (sin mirar: ${coma(D.modelo.metricas_validacion.brier_referencia, 4)})` }, svg);
}

function dibujarCurvas(k) {
  const c = D.curvas[k];
  document.querySelectorAll("#selector-fold button").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.fold) === k)));
  const pintar = (svg, a, b, ref, etiqueta, fy, ylim) => {
    vaciar(svg);
    const marco = [58, 332, 12, 180], ep = c.epocas, xmax = ep[ep.length - 1];
    const todos = [...c[a], ...c[b], ref];
    const [lo, hi] = ylim || [Math.min(...todos), Math.max(...todos)];
    const pad = (hi - lo) * 0.1, x = escala(1, xmax, marco[0], marco[1]), y = escala(lo - pad, hi + pad, marco[3], marco[2]);
    const yt = [lo, (lo + hi) / 2, hi];
    const xt = [1, Math.round(xmax / 2), xmax];
    ejes(svg, { x, y, xt, yt, fx: (v) => String(v), fy, xl: "época", yl: etiqueta, marco });
    s("line", { x1: marco[0], x2: marco[1], y1: y(ref), y2: y(ref), style: "stroke:var(--texto-3);stroke-dasharray:4 3" }, svg);
    s("line", { x1: x(c.elegida), x2: x(c.elegida), y1: marco[2], y2: marco[3], style: "stroke:var(--texto);stroke-width:1.5;opacity:.6" }, svg);
    for (const [clave, color] of [[a, "var(--nopcr)"], [b, "var(--pcr)"]]) {
      s("polyline", { points: ep.map((e, j) => `${x(e)},${y(c[clave][j])}`).join(" "), style: `fill:none;stroke:${color};stroke-width:2` }, svg);
    }
    ep.forEach((e, j) => s("rect", { x: x(e) - 6, y: marco[2], width: 12, height: marco[3] - marco[2], style: "fill:transparent",
      "data-tip": `<b>época ${e}</b>${e === c.elegida ? " (elegida)" : ""}<br>entrenamiento ${coma(c[a][j], 3)} · validación ${coma(c[b][j], 3)}` }, svg));
  };
  pintar($("g-perdida"), "perdida_train", "perdida_val", c.referencia_val, "pérdida", (v) => coma(v, 3));
  pintar($("g-auc"), "auc_train", "auc_val", 0.5, "AUC por paciente", (v) => coma(v, 2),
         [Math.min(0.4, ...c.auc_train, ...c.auc_val), Math.max(0.8, ...c.auc_train)]);
}

function dibujarArquitectura() {
  const svg = $("g-arquitectura"); vaciar(svg);
  const capas = [
    { n: "Entrada", f: "3 × 256 × 256", c: 3, l: 256, p: 0, d: "PRE, EARLY, LATE" },
    { n: "Recorte", f: "3 × 128 × 128", c: 3, l: 128, p: 0, d: "centro de la imagen" },
    { n: "Bloque 1", f: "16 × 64 × 64", c: 16, l: 64, p: 464, d: "conv 3×3 · BN · ReLU · pool" },
    { n: "Bloque 2", f: "32 × 32 × 32", c: 32, l: 32, p: 4672, d: "conv 3×3 · BN · ReLU · pool" },
    { n: "Bloque 3", f: "64 × 16 × 16", c: 64, l: 16, p: 18560, d: "conv 3×3 · BN · ReLU · pool" },
    { n: "Bloque 4", f: "128 × 8 × 8", c: 128, l: 8, p: 73984, d: "conv 3×3 · BN · ReLU · pool" },
    { n: "Media global", f: "128", c: 128, l: 1, p: 0, d: "un número por mapa" },
    { n: "Densa", f: "64", c: 64, l: 1, p: 8256, d: "ReLU · dropout 0,5" },
    { n: "Salida", f: "1", c: 1, l: 1, p: 65, d: "sigmoide → probabilidad" },
  ];
  const paso = 1100 / capas.length;
  capas.forEach((q, i) => {
    const cx = paso * i + paso / 2, alto = 24 + 110 * Math.sqrt(q.l / 256), ancho = 10 + 46 * Math.sqrt(q.c / 128);
    const y0 = 95 - alto / 2;
    s("rect", { x: cx - ancho / 2, y: y0, width: ancho, height: alto, rx: 3,
                style: q.p ? "fill:var(--acento);fill-opacity:.12;stroke:var(--acento);stroke-opacity:.5"
                           : "fill:var(--texto-3);fill-opacity:.06;stroke:var(--texto-3);stroke-opacity:.6;stroke-dasharray:3 3",
                "data-tip": `<b>${q.n}</b><br>${q.f}<br>${q.d}${q.p ? `<br>${q.p.toLocaleString("es-ES")} parámetros` : ""}` }, svg);
    if (i < capas.length - 1) s("path", { d: `M${cx + ancho / 2 + 10} 95 l${paso - ancho - 24} 0`, style: "stroke:var(--borde-fuerte);stroke-width:1.5;marker-end:url(#flecha)" }, svg);
    s("text", { class: "marca-texto", x: cx, y: 178, "text-anchor": "middle", texto: q.n }, svg);
    s("text", { x: cx, y: 194, "text-anchor": "middle", texto: q.f }, svg);
    s("text", { x: cx, y: 210, "text-anchor": "middle", style: "fill:var(--texto-3)", texto: q.p ? `${q.p.toLocaleString("es-ES")} p.` : "sin parámetros" }, svg);
  });
  const defs = s("defs", {}, svg), mk = s("marker", { id: "flecha", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
  s("path", { d: "M0 0 L10 5 L0 10 z", style: "fill:var(--borde-fuerte)" }, mk);
  $("arq-nota").textContent = `${D.modelo.entrenamiento.parametros.toLocaleString("es-ES")} parámetros · pasa el ratón por cada capa`;
}

function dibujarCohortes() {
  const svg = $("g-cohortes"); vaciar(svg);
  const nombres = { spy2: "I-SPY2", duke: "Duke", spy1: "I-SPY1" };
  const filas = Object.keys(nombres).map((k) => {
    const pares = D.validacion.pacientes.filter((q) => q[3] === k).map(([p, y]) => [p, y]);
    return { k, n: pares.length, pcr: pares.filter(([, y]) => y).length, auc: aucMannWhitney(pares) };
  });
  const marco = [70, 500, 6, 104], x = escala(0.4, 0.75, marco[0], marco[1]);
  const ge = s("g", { class: "eje" }, svg), gr = s("g", { class: "rejilla" }, svg);
  for (const v of [0.4, 0.5, 0.6, 0.7]) { s("line", { x1: x(v), x2: x(v), y1: marco[2], y2: marco[3] }, gr); s("text", { x: x(v), y: marco[3] + 16, "text-anchor": "middle", texto: coma(v, 1) }, ge); }
  s("line", { x1: x(0.5), x2: x(0.5), y1: marco[2], y2: marco[3], style: "stroke:var(--texto-3);stroke-dasharray:4 3" }, svg);
  filas.forEach((f, i) => {
    const y = marco[2] + 8 + i * 30;
    s("text", { x: marco[0] - 8, y: y + 12, "text-anchor": "end", texto: nombres[f.k] }, ge);
    s("rect", { x: x(0.4), y, width: Math.max(2, x(Math.min(f.auc, 0.75)) - x(0.4)), height: 18, rx: 2, style: "fill:var(--texto-2);opacity:.55",
                "data-tip": `<b>${nombres[f.k]}</b><br>${f.n} pacientes · ${f.pcr} con pCR (${pct(f.pcr / f.n)})<br>AUC ${coma(f.auc, 3)}` }, svg);
    s("text", { class: "marca-texto", x: x(Math.min(Math.max(f.auc, 0.4), 0.75)) + 6, y: y + 13, texto: coma(f.auc, 3) }, svg);
  });
}

function montarModelo() {
  const m = D.modelo, v = m.metricas_validacion, e = m.entrenamiento;
  const kpi = (k, val, sub) => `<div class="kpi"><div class="k">${k}</div><div class="v">${val}</div><div class="s">${sub}</div></div>`;
  $("kpis").innerHTML = kpi("AUC", coma(v.auc), `IC95 ${coma(v.auc_ic95[0], 2)}–${coma(v.auc_ic95[1], 2)}`) +
    kpi("Precisión media", coma(v.ap), `azar ${coma(PREVALENCIA, 2)}`) + kpi("Brier", coma(v.brier, 4), `referencia ${coma(v.brier_referencia, 4)}`) +
    kpi("Umbral", coma(m.umbral), "máxima F1 equilibrada") + kpi("Parámetros", e.parametros.toLocaleString("es-ES"), "por red") +
    kpi("Redes", String(m.redes.length), "3 semillas × 5 folds");
  dibujarArquitectura();
  ROC = curvaRoc();
  const des = $("u-deslizador");
  des.value = String(m.umbral);
  des.addEventListener("input", () => actualizarUmbral(Number(des.value)));
  document.querySelectorAll("[data-umbral]").forEach((b) => b.addEventListener("click", () => {
    const val = b.dataset.umbral === "eq" ? m.umbral : b.dataset.umbral === "des" ? m.umbral_desescalar : Number(b.dataset.umbral);
    des.value = String(val); actualizarUmbral(val);
  }));
  actualizarUmbral(m.umbral);
  dibujarCalibracion();
  $("selector-fold").innerHTML = D.curvas.map((c) => `<button data-fold="${c.fold}" aria-pressed="false">fold ${c.fold}</button>`).join("");
  document.querySelectorAll("#selector-fold button").forEach((b) => b.addEventListener("click", () => dibujarCurvas(Number(b.dataset.fold))));
  dibujarCurvas(0);
  const ep = e.epocas_elegidas;
  const fila = (k, val) => `<dt>${k}</dt><dd>${val}</dd>`;
  $("ficha-entrenamiento").innerHTML = fila("Optimizador", `${e.optimizador} · tasa ${String(e.lr).replace(".", ",")}`) + fila("Lote", e.lote) +
    fila("Pérdida", "entropía cruzada binaria (sin ponderar)") +
    fila("Parada", `AUC de validación en media móvil de 3 épocas · paciencia 10 · máximo ${e.epocas_max}`) +
    fila("Épocas elegidas", `${Math.min(...ep)} a ${Math.max(...ep)} (media ${coma(mediaDe(ep), 1)})`) + fila("Semillas", e.semillas.join(", ")) +
    fila("Calibración", `Platt · a = ${coma(m.calibracion.a, 3)}, b = ${coma(m.calibracion.b, 3)}`) +
    fila("Entrenado con", `${e.dispositivo_entrenamiento} · ${coma(e.segundos_por_epoca, 1)} s/época`) + fila("PyTorch", e.version_torch);
  dibujarCohortes();
}

// ------------------------------------------------------------------------------------------------ vista Verificacion
function montarVerificacion() {
  const fila = (k, val) => `<dt>${k}</dt><dd class="mono">${val}</dd>`;
  $("ficha-checksum").innerHTML = fila("Checksum PyTorch", D.modelo.checksum_pytorch) +
    fila("Checksum web esperado", W ? W.checksum_web : "—") + fila("Checksum web calculado", '<span id="cs-calc">—</span>') +
    fila("ONNX frente a PyTorch", W ? `${W.onnx_frente_a_pytorch.toExponential(1)} (logit)` : "—");
  if (DEMO) {
    $("sello").querySelector("h3").textContent = "No disponible en la maqueta";
    $("sello").querySelector(".texto-suave").textContent = "La verificación necesita los modelos: funciona en la página publicada.";
  }
}

const etiquetaClase = (c) => (c === "pCR" ? "pCR = 1" : "pCR = 0");
async function verificar() {
  const tabla = $("t-verificacion"); $("verificar").disabled = true;
  const sello = $("sello"); sello.className = "sello"; sello.querySelector(".icono").textContent = "…"; sello.querySelector("h3").textContent = "Verificando…";
  const filas = []; let peor = 0, ok = true;
  for (const e of W.ejemplos) {
    try {
      const f = {}; for (const k of FASES) f[k] = leerFase(await bajar(e.rutas[k]), k).x;
      const r = await predecirCorte(f), dif = Math.abs(r.probabilidad - e.probabilidad_python), bien = dif < TOLERANCIA && r.clase === e.clase_python;
      peor = Math.max(peor, dif); ok = ok && bien;
      filas.push(`<tr><td>${e.sample_id}</td><td class="num">${e.probabilidad_python.toFixed(6)}</td><td class="num">${r.probabilidad.toFixed(6)}</td><td class="num">${dif.toExponential(1)}</td><td>${e.clase_python === r.clase ? etiquetaClase(r.clase) : `${etiquetaClase(e.clase_python)} / ${etiquetaClase(r.clase)}`}</td><td class="${bien ? "bien" : "mal"}">${bien ? "✓" : "✗"}</td></tr>`);
    } catch (err) { ok = false; filas.push(`<tr><td>${e.sample_id}</td><td colspan="5" class="mal">${err.message}</td></tr>`); }
  }
  tabla.hidden = false;
  tabla.innerHTML = `<thead><tr><th>Corte</th><th>Python</th><th>Esta página</th><th>Diferencia</th><th>Clase</th><th></th></tr></thead><tbody>${filas.join("")}</tbody>`;
  sello.className = `sello ${ok ? "bien" : "mal"}`;
  sello.querySelector(".icono").textContent = ok ? "✓" : "✗";
  sello.querySelector("h3").textContent = ok ? "La página reproduce el pipeline de Python" : "Hay diferencias con Python";
  sello.querySelector(".texto-suave").textContent = `${W.ejemplos.length} cortes · diferencia máxima de probabilidad ${peor.toExponential(1)} (tolerancia ${TOLERANCIA.toExponential(0)})`;
  $("verificar").disabled = false;
}

// ------------------------------------------------------------------------------------------------ arranque
function cambiarVista(v) {
  document.querySelectorAll(".vista").forEach((s_) => s_.classList.toggle("activa", s_.id === `vista-${v}`));
  document.querySelectorAll(".menu button").forEach((b) => { if (b.dataset.vista === v) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
  try { history.replaceState(null, "", `#${v}`); } catch (e) {}
  window.scrollTo({ top: 0 });
}

function iniciar() {
  if (!D) { document.body.insertAdjacentHTML("afterbegin", '<div class="franja franja-demo">Falta datos/datos.js: ejecuta python datos_web.py</div>'); return; }
  $("franja-demo").hidden = !DEMO;
  document.querySelectorAll(".menu button").forEach((b) => b.addEventListener("click", () => cambiarVista(b.dataset.vista)));
  const inicial = location.hash.slice(1); if (["predecir", "modelo", "verificacion", "acerca"].includes(inicial)) cambiarVista(inicial);
  $("tema").addEventListener("click", () => {
    const claro = document.documentElement.dataset.theme !== "light";
    if (claro) document.documentElement.dataset.theme = "light"; else delete document.documentElement.dataset.theme;
    try { localStorage.setItem("tema", claro ? "light" : ""); } catch (e) {}
  });
  document.querySelectorAll('.ranura input[type="file"]').forEach((input) => {
    const fase = input.dataset.fase, ranura = input.closest(".ranura");
    input.addEventListener("change", async () => { const f = input.files[0]; if (!f) { ponerFichero(fase, null, null); return; } ponerFichero(fase, await f.arrayBuffer(), f.name); });
    ["dragenter", "dragover"].forEach((t) => input.addEventListener(t, () => ranura.classList.add("encima")));
    ["dragleave", "drop"].forEach((t) => input.addEventListener(t, () => ranura.classList.remove("encima")));
  });
  $("predecir").addEventListener("click", predecir);
  $("limpiar").addEventListener("click", limpiar);
  document.querySelectorAll("[data-fase-visor]").forEach((b) => b.addEventListener("click", () => { if (estado.animando) { cancelAnimationFrame(estado.animando); estado.animando = null; } mostrarFase(b.dataset.faseVisor); }));
  $("play").addEventListener("click", animarSecuencia);
  $("ver-recorte").addEventListener("change", () => { if (estado.fases) mostrarFase(estado.faseVisor); });
  $("descargar").addEventListener("click", descargarHistorial);
  $("vaciar").addEventListener("click", () => { estado.historial = []; $("historial").innerHTML = '<tr><td colspan="8" class="texto-tenue">Aún no hay predicciones en esta sesión.</td></tr>'; $("descargar").disabled = true; $("vaciar").disabled = true; });
  $("verificar").addEventListener("click", verificar);
  $("r-umbral").textContent = `${coma(D.modelo.umbral, 3)} (equilibrado)`;
  $("r-dispositivo").textContent = DEMO ? "maqueta (sin modelos)" : "navegador · WebAssembly";
  $("r-checksum").textContent = DEMO ? "— (maqueta)" : "calculando…";
  $("r-checksum-pt").textContent = D.modelo.checksum_pytorch;
  cineticaVacia();
  resultadoInicial();
  montarGaleria();
  montarModelo();
  montarVerificacion();
  cargarModelo();
}

iniciar();
