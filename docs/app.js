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
 *
 * Datos: window.DATOS (datos_web.py: metricas, validacion, test, curvas), window.WEB (exportar_web.py:
 * modelos y cortes de la Verificacion) y window.GALERIA (galeria_web.py: cortes de la galeria). MODO MAQUETA: abierta como fichero (o con ?demo) no se cargan los modelos y las predicciones son
 * inventadas, avisado con una franja; todo lo demas usa los datos reales.
 */

const D = window.DATOS;
const W = window.WEB || null;
// Galería: cortes de entrenamiento donde el tumor se ve bien (galeria_web.py); si faltan, los de la Verificación
const EJEMPLOS = window.GALERIA && window.GALERIA.ejemplos && window.GALERIA.ejemplos.length ? window.GALERIA.ejemplos : (W ? W.ejemplos : []);
const DEMO = location.protocol === "file:" || new URLSearchParams(location.search).has("demo");
const LADO = 256, RECORTE = 128, INI = (LADO - RECORTE) / 2, TAM_MAX = 5 * 1024 * 1024, TOLERANCIA = 1e-4;
const FASES = ["PRE", "EARLY", "LATE"];
const SEMILLAS = [42, 1, 2];
const SIN_MOVIMIENTO = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
const ANOTACION = "#ffd23f";      // color de las anotaciones sobre la imagen (zona de interes)
const MODELOS = ["recorte", "clinico", "combinado", "solo_cohorte"];
const NOMBRE_MODELO = { recorte: "CNN solo imagen", clinico: "solo datos clínicos", combinado: "imagen y datos clínicos",
                        solo_cohorte: "solo cohorte" };
const COLOR_MODELO = { recorte: "var(--m-cnn)", clinico: "var(--m-clin)", combinado: "var(--m-comb)", solo_cohorte: "var(--m-coh)" };
const NOMBRE_COHORTE = { duke: "Duke", spy1: "I-SPY1", spy2: "I-SPY2" };

const estado = {
  sesiones: [], checksumWeb: null, listo: false,
  fases: null,                 // {PRE, EARLY, LATE}: Float32Array de la muestra cargada
  buffers: { PRE: null, EARLY: null, LATE: null }, nombres: { PRE: null, EARLY: null, LATE: null },
  leidas: { PRE: null, EARLY: null, LATE: null },   // Float32Array de cada fase que se ha leido bien
  ejemploActivo: null,         // indice del ejemplo cargado (o null si se subio a mano)
  faseVisor: "EARLY", animando: null, historial: [],
  cortina: 0.5, par: ["PRE", "EARLY"], arrastrando: false, puntero: null,
  foco: 0,                     // durante el cálculo, cuánto se oscurece lo que queda fuera de la zona de interés (0 a 1)
  lote: [], loteActivo: null, procesando: false, conjunto: "test", portada: null,
};

class EntradaNoValida extends Error {
  constructor(mensaje, comprobaciones) { super(mensaje); this.comprobaciones = comprobaciones || []; }
}

// ------------------------------------------------------------------------------------------------ utilidades
const $ = (id) => document.getElementById(id);
const coma = (x, d = 3) => Number(x).toFixed(d).replace(".", ",");
const pct = (x, d = 0) => `${(100 * x).toFixed(d).replace(".", ",")} %`;
const conSigno = (x, d = 3) => `${x >= 0 ? "+" : "−"}${coma(Math.abs(x), d)}`;
const sigmoide = (z) => 1 / (1 + Math.exp(-z));
const limitar = (v, a, b) => Math.min(Math.max(v, a), b);
const esperar = (ms) => new Promise((r) => setTimeout(r, ms));
const miles = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
const tasaLegible = (lr) => {   // 3e-05 -> "3·10⁻⁵"
  const [m, e] = Number(lr).toExponential().split("e"), sup = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return `${m.replace(".", ",")}·10${String(Number(e)).split("").map((c) => sup[c]).join("")}`;
};
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
const MASCARA_CENTRO = (() => {
  const m = new Uint8Array(LADO * LADO);
  for (let y = INI; y < INI + RECORTE; y++) for (let x = INI; x < INI + RECORTE; x++) m[y * LADO + x] = 1;
  return m;
})();

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
  c.push({ t: `${img.width} × ${img.height}`, ok: medidas });
  if (!medidas) throw new EntradaNoValida(`${nombre}: debe medir ${LADO} × ${LADO} píxeles (mide ${img.width} × ${img.height}).`, c);
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

// Orden de las fases: en el tejido (PRE > 0,1), PRE es la fase más oscura en el 99,5 % de los cortes de entrenamiento
// (D.calidad, de datos_web.py). Si no lo es, seguramente alguna imagen está en la casilla equivocada. EARLY y LATE no
// tienen un orden fijo entre sí (LATE brilla más en el 79 %), así que intercambiarlas no se puede detectar.
function ordenFases(f) {
  const tejido = new Uint8Array(f.PRE.length);
  let n = 0;
  for (let i = 0; i < f.PRE.length; i++) if (f.PRE[i] > 0.1) { tejido[i] = 1; n++; }
  if (n < 100) return null;
  const pre = mediaDe(f.PRE, tejido);
  return { menos: ["EARLY", "LATE"].filter((k) => mediaDe(f[k], tejido) <= pre) };
}
const pctPreOscura = () => (D.calidad ? pct(D.calidad.pre_mas_oscura, 1) : "casi todos");
function avisoOrden(f) {
  const el = $("aviso-orden"), o = f ? ordenFases(f) : null;
  el.hidden = !o;
  if (!o) return;
  el.classList.toggle("mal", o.menos.length > 0);
  el.innerHTML = o.menos.length
    ? `<span><b>¿Está cada imagen en su casilla?</b> En el tejido, ${o.menos.join(" y ")} ${o.menos.length > 1 ? "brillan" : "brilla"} menos que PRE, ` +
      `y PRE es la fase más oscura en el ${pctPreOscura()} de los cortes de entrenamiento. Se puede predecir igualmente.</span>`
    : `<span>PRE es la fase más oscura, como en el ${pctPreOscura()} de los cortes de entrenamiento.</span>`;
}
// Brillo medio de la zona de interés (lo único que mira la red) en cada fase
function brilloZona(f) {
  const b = {};
  for (const k of FASES) {
    let s_ = 0;
    for (let y = INI; y < INI + RECORTE; y++) for (let x = INI; x < INI + RECORTE; x++) s_ += f[k][y * LADO + x];
    b[k] = s_ / (RECORTE * RECORTE);
  }
  return b;
}

// Control de calidad para el informe: las comprobaciones de predecir.avisos (también cuando pasan), el orden de las
// fases y el brillo frente al entrenamiento
function controlCalidad(f) {
  const tejido = new Uint8Array(f.PRE.length);
  let n = 0;
  for (let i = 0; i < f.PRE.length; i++) if (f.PRE[i] > 0.1) { tejido[i] = 1; n++; }
  const lista = [{ ok: true, texto: "Tres PNG de 256 × 256 en escala de grises, sin valores no finitos" }];
  if (n < 100) lista.push({ ok: false, texto: "La imagen PRE es casi toda negra: ¿es un corte de resonancia de mama?" });
  else {
    const { menos } = ordenFases(f);
    lista.push(menos.length
      ? { ok: false, texto: `En el tejido, ${menos.join(" y ")} no ${menos.length > 1 ? "son más brillantes" : "es más brillante"} que PRE: ¿está cada imagen en su casilla?` }
      : { ok: true, texto: `En el tejido, PRE es la fase más oscura, como en el ${pctPreOscura()} de los cortes de entrenamiento` });
  }
  const uniformes = FASES.filter((k) => desviacion(f[k]) < 1e-3), blancas = FASES.filter((k) => mediaDe(f[k]) > 0.8);
  lista.push(uniformes.length ? { ok: false, texto: `La fase ${uniformes.join(" y ")} es una imagen prácticamente uniforme` }
                              : { ok: true, texto: "Ninguna fase es una imagen uniforme" });
  lista.push(blancas.length ? { ok: false, texto: `La fase ${blancas.join(" y ")} es casi toda blanca: la predicción no es fiable` }
                            : { ok: true, texto: "Ninguna fase está saturada en blanco" });
  const rango = D.calidad && D.calidad.brillo_zona;
  if (rango) {
    const b = brilloZona(f), fuera = FASES.filter((k) => b[k] < rango[k][0] || b[k] > rango[k][1]);
    lista.push(fuera.length
      ? { ok: false, texto: "El brillo de la zona de interés se sale de lo habitual en el entrenamiento: " +
          fuera.map((k) => `${k} ${coma(b[k], 2)} (lo habitual, de ${coma(rango[k][0], 2)} a ${coma(rango[k][1], 2)})`).join("; ") +
          ". El modelo es sensible al brillo: la predicción es menos fiable" }
      : { ok: true, texto: "El brillo de la zona de interés está dentro de lo habitual en el entrenamiento (el 99 % de los cortes)" });
  }
  return lista;
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
    ponerEstado("Maqueta, sin modelos", "", 1);
    estado.listo = true;
    actualizarBotonPredecir();
    return;
  }
  if (!W) { ponerEstado("Faltan los modelos: ejecuta exportar_web.py en el PC", "fallo", 0); $("r-checksum").textContent = "—"; return; }
  try {
    ort.env.wasm.numThreads = 1;     // GitHub Pages no permite hilos de WebAssembly; resultado reproducible
    const buffers = [];
    for (let i = 0; i < W.modelos.length; i++) {
      ponerEstado(`Cargando la red ${i + 1} de ${W.modelos.length}`, "", i / W.modelos.length);
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
    ponerEstado(ok ? `Modelo listo: ${W.modelos.length} redes, checksum ${estado.checksumWeb}` : `Checksum distinto (${estado.checksumWeb})`, ok ? "listo" : "fallo", 1);
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

// alRed(i, p) se llama al terminar cada red; pausa (ms) solo para que la animacion del proceso se pueda seguir
async function predecirCorte(f, { alRed = null, pausa = 0 } = {}) {
  const { umbral, calibracion } = D.modelo;
  const probs = [];
  let ms = 0;
  if (DEMO) {
    const base = 0.24 + 0.22 * hash01(f);
    for (let i = 0; i < 15; i++) {
      const p = limitar(base + 0.05 * Math.sin(7.3 * i + 10 * base), 0.02, 0.95);
      probs.push(p);
      if (alRed) alRed(i, p);
      await esperar(pausa || 8);
    }
  } else {
    const x = new Float32Array(3 * LADO * LADO);
    FASES.forEach((k, j) => x.set(f[k], j * LADO * LADO));     // canales: PRE, EARLY, LATE
    const tensor = new ort.Tensor("float32", x, [1, 3, LADO, LADO]);
    for (let i = 0; i < estado.sesiones.length; i++) {
      const t0 = performance.now();
      const salida = await estado.sesiones[i].run({ imagen: tensor });
      ms += performance.now() - t0;
      const p = sigmoide(salida.logit.data[0]);
      probs.push(p);
      if (alRed) alRed(i, p);
      if (pausa) await esperar(pausa);
    }
  }
  let m = probs.reduce((a, b) => a + b, 0) / probs.length;
  m = limitar(m, 1e-6, 1 - 1e-6);
  const p = sigmoide(calibracion.a * Math.log(m / (1 - m)) + calibracion.b);
  const clase = p >= umbral ? "pCR" : "no pCR";
  return { probabilidad: p, clase, media: m, probs, segundos: ms / 1000 };
}

// La misma predicción con la imagen un 10 % más oscura y un 10 % más clara, como en analisis/diagnostico_estabilidad.py
// (allí cambian de clase el 13-16 % de las pacientes de validación). Si alguna cruza el umbral, la decisión es ajustada.
async function predecirBrillo(f, r) {
  if (DEMO) { const d = 0.01 + 0.025 * hash01(f); return { oscura: limitar(r.probabilidad - d, 0, 1), clara: limitar(r.probabilidad + 0.9 * d, 0, 1) }; }
  const variante = (k) => Object.fromEntries(FASES.map((fase) => [fase, f[fase].map((v) => v * k)]));
  const oscura = await predecirCorte(variante(0.9)), clara = await predecirCorte(variante(1.1));
  return { oscura: oscura.probabilidad, clara: clara.probabilidad };
}

// ------------------------------------------------------------------------------------------------ imagenes
const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function colorInferno(t) {
  const x = limitar(t, 0, 1) * (INFERNO.length - 1);
  const i = Math.min(Math.floor(x), INFERNO.length - 2), r = x - i;
  return INFERNO[i].map((c, k) => Math.round(c + r * (INFERNO[i + 1][k] - c)));
}
function pintarGris(ctx, datos, x0 = 0) {
  const img = ctx.createImageData(LADO, LADO);
  for (let i = 0, j = 0; i < datos.length; i++, j += 4) {
    const v = Math.round(limitar(datos[i], 0, 1) * 255);
    img.data[j] = img.data[j + 1] = img.data[j + 2] = v; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, x0, 0);
}
// Realce = EARLY - PRE (la operacion que, segun la guia, hace visible el tumor). El color se escala con el percentil 99:
// si se escalara con el pixel mas brillante (a veces un vaso o el borde de la mama), casi toda la imagen saldria negra.
function pintarRealce(ctx, pre, early) {
  const img = ctx.createImageData(LADO, LADO), r = new Float32Array(pre.length);
  for (let i = 0; i < pre.length; i++) r[i] = Math.max(early[i] - pre[i], 0);
  const positivos = Array.from(r).filter((v) => v > 0.01).sort((a, b) => a - b);
  const max = Math.max(positivos.length ? positivos[Math.floor(0.99 * (positivos.length - 1))] : 0, 1e-3);
  for (let i = 0, j = 0; i < r.length; i++, j += 4) {
    const [a, b, c] = colorInferno(r[i] / max);
    img.data[j] = a; img.data[j + 1] = b; img.data[j + 2] = c; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function pintarComparar(ctx) {    // izquierda: la primera fase del par; derecha: la segunda
  const [a, b] = estado.par, A = estado.fases[a], B = estado.fases[b], corte = Math.round(estado.cortina * LADO);
  const img = ctx.createImageData(LADO, LADO);
  for (let y = 0; y < LADO; y++) for (let x = 0; x < LADO; x++) {
    const i = y * LADO + x, j = 4 * i, v = Math.round(limitar(x < corte ? A[i] : B[i], 0, 1) * 255);
    img.data[j] = img.data[j + 1] = img.data[j + 2] = v; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function pintarRecorte(ctx) {
  if (!$("ver-recorte").checked) return;
  ctx.save(); ctx.strokeStyle = ANOTACION; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
  ctx.strokeRect(INI + 0.75, INI + 0.75, RECORTE - 1.5, RECORTE - 1.5); ctx.restore();
}
function mezcla(a, b, t) { const r = new Float32Array(a.length); for (let i = 0; i < a.length; i++) r[i] = a[i] + t * (b[i] - a[i]); return r; }

function nombreMuestra() {
  if (estado.loteActivo !== null && estado.lote[estado.loteActivo]) return estado.lote[estado.loteActivo].clave;
  if (estado.ejemploActivo !== null) return DEMO ? `Fantasma ${estado.ejemploActivo + 1}` : EJEMPLOS[estado.ejemploActivo].sample_id;
  const n = estado.nombres.EARLY || estado.nombres.PRE || "";
  return n.replace(/\.png$/i, "").replace(/[_\-\s.]?(PRE|EARLY|LATE)$/i, "").replace(/ \((PRE|EARLY|LATE)\)$/, "") || "Muestra";
}

const DESCRIPCION = { PRE: "Antes del contraste", EARLY: "Contraste temprano", LATE: "Contraste tardío", REALCE: "EARLY menos PRE",
                      COMPARAR: "", TRES: "En el orden del modelo" };
const NOMBRE_VISTA = { REALCE: "Realce", TRES: "Las tres fases" };
const HUECO = 8;     // separación entre las tres imágenes de "Las tres", en píxeles del lienzo
// Anotaciones de las esquinas del visor, como en una estacion de radiologia
function ponerAnotaciones(etiqueta, titulo = null) {
  if (etiqueta === "COMPARAR") {
    $("ov-tl").innerHTML = `<b>${estado.par[0]}</b><br>${DESCRIPCION[estado.par[0]]}`;
    $("ov-tr").innerHTML = `<b>${estado.par[1]}</b><br>${DESCRIPCION[estado.par[1]]}`;
  } else {
    $("ov-tl").innerHTML = `<b>${titulo || NOMBRE_VISTA[etiqueta] || etiqueta}</b><br>${titulo ? "Llegada del contraste" : DESCRIPCION[etiqueta]}`;
    $("ov-tr").innerHTML = etiqueta === "TRES" ? "256 × 256 cada una<br>8 bits por píxel" : "256 × 256<br>8 bits por píxel";
  }
  const lineas = [];
  if ($("ver-recorte").checked) lineas.push('<span class="anot">Zona de interés</span>', "128 × 128");
  $("ov-bl").innerHTML = lineas.join("<br>");
  $("ov-br").innerHTML = `${escapar(nombreMuestra())}<br>Un corte`;
}
function quitarAnotaciones() { ["ov-tl", "ov-tr", "ov-bl", "ov-br"].forEach((k) => { $(k).innerHTML = ""; }); }

function dibujarVisor(datos, etiqueta, titulo = null) {
  const lienzo = $("visor"), tres = etiqueta === "TRES", ancho = tres ? 3 * LADO + 2 * HUECO : LADO;
  if (lienzo.width !== ancho) lienzo.width = ancho;          // "Las tres": un lienzo con las tres imágenes en fila
  const ctx = lienzo.getContext("2d"), origenes = tres ? FASES.map((_, j) => j * (LADO + HUECO)) : [0];
  if (tres) {
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, ancho, LADO);
    FASES.forEach((k, j) => pintarGris(ctx, estado.fases[k], origenes[j]));
    ctx.fillStyle = "#3a424d";
    for (let j = 1; j < FASES.length; j++) ctx.fillRect(origenes[j] - HUECO / 2 - 1, 0, 2, LADO);
  } else if (etiqueta === "REALCE") pintarRealce(ctx, estado.fases.PRE, estado.fases.EARLY);
  else if (etiqueta === "COMPARAR") pintarComparar(ctx);
  else pintarGris(ctx, datos || estado.fases[etiqueta]);
  for (const x0 of origenes) {
    ctx.save(); ctx.translate(x0, 0);
    if (estado.foco > 0) {      // mientras se calcula: solo cuenta la zona de interés, el resto se oscurece
      ctx.fillStyle = `rgba(0, 0, 0, ${0.68 * estado.foco})`;
      ctx.fillRect(0, 0, LADO, INI); ctx.fillRect(0, INI + RECORTE, LADO, LADO - INI - RECORTE);
      ctx.fillRect(0, INI, INI, RECORTE); ctx.fillRect(INI + RECORTE, INI, LADO - INI - RECORTE, RECORTE);
    }
    pintarRecorte(ctx);
    ctx.restore();
  }
  ponerAnotaciones(etiqueta, titulo);
  rotulosTres(tres);
  const partes = [];
  if (etiqueta === "REALCE") partes.push('<span><span class="barra-realce"></span> Realce: cuánto más brilla cada punto en EARLY que en PRE (oscuro, nada; claro, mucho). Lo que capta el contraste, como el tumor, se ilumina.</span>');
  if (etiqueta === "COMPARAR") partes.push("<span>Arrastra sobre la imagen para mover la cortina.</span>");
  if (tres) partes.push("<span>Las tres fases, en el orden en que las recibe el modelo. Con el contraste, el tumor se ilumina en EARLY y LATE; en PRE apenas destaca.</span>");
  $("leyenda-visor").innerHTML = partes.join("");
  $("cortina").hidden = etiqueta !== "COMPARAR";
  const imagen = rectanguloImagen(), cortina = $("cortina");
  cortina.style.left = `${imagen.x + estado.cortina * imagen.lado}px`;
  cortina.style.top = `${imagen.y}px`; cortina.style.height = `${imagen.lado}px`;
  if (estado.puntero) dibujarLupa(estado.puntero);
}
// Rótulo de cada fase en la esquina de su imagen (si la imagen es estrecha, solo el nombre de la fase)
function rotulosTres(tres) {
  const caja = $("ov-tres");
  caja.hidden = !tres;
  if (!tres) return;
  const im = rectanguloImagen(), w = $("visor").width, panel = (im.ancho * LADO) / w;
  [...caja.children].forEach((el, j) => {
    el.innerHTML = panel >= 170 ? `<b>${FASES[j]}</b><br>${DESCRIPCION[FASES[j]]}` : `<b>${FASES[j]}</b>`;
    el.style.left = `${im.x + (j * (LADO + HUECO) * im.ancho) / w + 8}px`;
    el.style.top = `${im.y + 6}px`;
  });
}
function mostrarFase(fase) {
  estado.faseVisor = fase;
  document.querySelectorAll("[data-fase-visor]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.faseVisor === fase)));
  $("par-comparar").hidden = fase !== "COMPARAR";
  $("marco-visor").classList.toggle("comparando", fase === "COMPARAR");
  if (estado.fases) dibujarVisor(null, fase);
}
function pararSecuencia() {
  if (estado.animando) cancelAnimationFrame(estado.animando);
  estado.animando = null;
  $("play").classList.remove("activo");
}
function animarSecuencia() {
  if (!estado.fases) return;
  if (estado.animando) { pararSecuencia(); mostrarFase(estado.faseVisor); return; }
  const tramos = [["PRE", "EARLY"], ["EARLY", "LATE"]], dur = 1300, pausa = 450, inicio = performance.now();
  const total = 2 * (tramos.length * (dur + pausa));
  $("play").classList.add("activo");
  const paso = () => {
    const ahora = performance.now();
    const t = Math.max(0, ahora - inicio) % (tramos.length * (dur + pausa));
    const k = Math.floor(t / (dur + pausa)), dentro = t - k * (dur + pausa);
    const [a, b] = tramos[k];
    const u = Math.min(dentro / dur, 1), suave = u * u * (3 - 2 * u);
    const titulo = suave < 0.02 ? a : suave > 0.98 ? b : `${a} → ${b}`;
    dibujarVisor(mezcla(estado.fases[a], estado.fases[b], suave), suave < 0.5 ? a : b, titulo);
    if (ahora - inicio < total) estado.animando = requestAnimationFrame(paso);
    else { estado.animando = null; $("play").classList.remove("activo"); mostrarFase("LATE"); }
  };
  estado.animando = requestAnimationFrame(paso);
}
function miniatura(destino, datos) {
  const c = document.createElement("canvas"); c.width = c.height = LADO;
  pintarGris(c.getContext("2d"), datos);
  destino.innerHTML = ""; destino.appendChild(c);
  return c;
}

// La imagen se dibuja lo mas grande posible y centrada dentro de la ventana del visor (cuadrada, o las tres en fila):
// su posicion dentro de la ventana y su tamaño en pantalla
function rectanguloImagen() {
  const lienzo = $("visor"), caja = lienzo.getBoundingClientRect(), r = lienzo.width / lienzo.height;
  const ancho = Math.min(caja.width, caja.height * r), alto = ancho / r;
  return { caja, x: (caja.width - ancho) / 2, y: (caja.height - alto) / 2, ancho, alto, lado: alto };
}

// ---- lupa: amplia x3 lo que se ve en el visor ----
function dibujarLupa(ev) {
  const lupa = $("lupa"), visor = $("visor");
  if (!$("ver-lupa").checked || !estado.fases || !ev) { lupa.hidden = true; return; }
  const { caja, x: ox, y: oy, ancho, alto } = rectanguloImagen();
  const x = ev.clientX - caja.left, y = ev.clientY - caja.top, ix = x - ox, iy = y - oy;
  if (ix < 0 || iy < 0 || ix > ancho || iy > alto) { lupa.hidden = true; return; }
  lupa.hidden = false;
  const tam = lupa.offsetWidth || 150, zoom = 3, dpr = window.devicePixelRatio || 1;
  lupa.style.left = `${x - tam / 2}px`; lupa.style.top = `${y - tam / 2}px`;
  if (lupa.width !== Math.round(tam * dpr)) { lupa.width = lupa.height = Math.round(tam * dpr); }
  const k = visor.height / alto, lado = (tam / zoom) * k;
  const ctx = lupa.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, lupa.width, lupa.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(visor, ix * k - lado / 2, iy * k - lado / 2, lado, lado, 0, 0, lupa.width, lupa.height);
}

// ---- comparador: la cortina se arrastra con el raton o el dedo ----
function moverCortina(ev) {
  const { caja, x, lado } = rectanguloImagen();
  estado.cortina = limitar((ev.clientX - caja.left - x) / lado, 0, 1);
  dibujarVisor(null, "COMPARAR");
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
    const tip = `<b>${coma(a, 2)} a ${coma(a + w, 2)}</b><br>${n0 + n1} pacientes, ${n1} con pCR${n0 + n1 ? ` (${pct(n1 / (n0 + n1))})` : ""}`;
    if (n0) s("rect", { x: xa, width: wa, y: y(n0), height: y(0) - y(n0), rx: 1, style: "fill:var(--nopcr)", "data-tip": tip }, svg);
    if (n1) s("rect", { x: xa, width: wa, y: y(n0 + n1), height: Math.max(1, y(n0) - y(n0 + n1) - (n0 ? 1.5 : 0)), rx: 1, style: "fill:var(--pcr)", "data-tip": tip }, svg);
  });
  if (umbral !== undefined) {
    s("line", { x1: x(umbral), x2: x(umbral), y1: marco[2], y2: marco[3], style: "stroke:var(--texto-2);stroke-dasharray:4 3;stroke-width:1.5" }, svg);
  }
  if (marca !== undefined) {
    const xm = x(limitar(marca, x0, x1));
    s("line", { x1: xm, x2: xm, y1: marco[2] - 2, y2: marco[3], style: "stroke:var(--texto);stroke-width:2.5" }, svg);
    s("circle", { cx: xm, cy: marco[2] + 2, r: 5, style: "fill:var(--texto)" }, svg);
    s("text", { class: "marca-texto", x: xm + (xm > ancho * 0.7 ? -8 : 8), y: marco[2] + 6, "text-anchor": xm > ancho * 0.7 ? "end" : "start", texto: "esta muestra" }, svg);
  }
}

// Escala de probabilidad, como un rango de referencia en un informe: barra de 0 a 1 (azul hasta el umbral, naranja
// después) y, encima, cómo se reparten las 1.097 pacientes de validación; la marca blanca es la muestra
function medidor(p, u) {
  const svg = $("g-medidor"); vaciar(svg);
  const x0 = 8, x1 = 352, yb = 56, x = escala(0, 1, x0, x1), w = 0.02, nb = Math.round(1 / w);
  const bins = Array.from({ length: nb }, () => [0, 0]);
  for (const [q, y] of D.validacion.pacientes) bins[Math.min(nb - 1, Math.floor(q / w))][y]++;
  const max = Math.max(...bins.map(([a, b]) => a + b)), h = escala(0, max, 0, 34);
  bins.forEach(([n0, n1], k) => {
    const xa = x(k * w) + 0.5, wa = x(w) - x(0) - 1;
    if (n0) s("rect", { x: xa, width: wa, y: yb - 8 - h(n0), height: h(n0), style: "fill:var(--nopcr);opacity:.55" }, svg);
    if (n1) s("rect", { x: xa, width: wa, y: yb - 8 - h(n0 + n1), height: h(n1), style: "fill:var(--pcr);opacity:.55" }, svg);
  });
  s("line", { x1: x0, x2: x(u), y1: yb, y2: yb, style: "stroke:var(--nopcr);stroke-width:6;stroke-linecap:round" }, svg);
  s("line", { x1: x(u), x2: x1, y1: yb, y2: yb, style: "stroke:var(--pcr);stroke-width:6;stroke-linecap:round" }, svg);
  s("line", { x1: x(u), x2: x(u), y1: yb - 46, y2: yb + 8, style: "stroke:var(--texto-2);stroke-width:1.2;stroke-dasharray:3 3" }, svg);
  s("text", { x: x0, y: yb + 24, texto: "0" }, svg);
  s("text", { x: x1, y: yb + 24, "text-anchor": "end", texto: "1" }, svg);
  s("text", { x: x(u), y: yb + 24, "text-anchor": "middle", texto: `umbral ${coma(u, 3)}` }, svg);
  const g = s("g", { transform: `translate(${x0} 0)` }, svg);
  s("line", { x1: 0, x2: 0, y1: yb - 48, y2: yb + 9, style: "stroke:var(--texto);stroke-width:2" }, g);
  s("circle", { cx: 0, cy: yb - 48, r: 4, style: "fill:var(--texto)" }, g);
  animar(900, (k) => g.setAttribute("transform", `translate(${x(p * k)} 0)`));
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
    s("circle", { cx: x(p), cy: fila(fold) + (sem - 1) * 5, r: 4.5, style: "fill:var(--texto-2);stroke:var(--panel);stroke-width:2",
                  "data-tip": `<b>fold ${fold}, semilla ${SEMILLAS[sem]}</b><br>probabilidad ${coma(p, 3)}` }, svg);
  });
}

function graficaCinetica(f) {
  const svg = $("g-cinetica"); vaciar(svg);
  const tejido = new Uint8Array(LADO * LADO);
  for (let i = 0; i < tejido.length; i++) if (f.PRE[i] > 0.1) tejido[i] = 1;
  const series = [
    { nombre: "zona de interés", color: "var(--serie-3)", v: FASES.map((k) => mediaDe(f[k], MASCARA_CENTRO)) },
    { nombre: "todo el tejido", color: "var(--texto-3)", v: FASES.map((k) => mediaDe(f[k], tejido)) },
  ];
  const ymax = Math.max(...series.flatMap((q) => q.v)) * 1.18 || 1;
  const marco = [46, 350, 14, 160];
  const x = (k) => marco[0] + 30 + k * ((marco[1] - marco[0] - 60) / 2), y = escala(0, ymax, marco[3], marco[2]);
  const gr = s("g", { class: "rejilla" }, svg), ge = s("g", { class: "eje" }, svg);
  for (const v of [0, ymax / 2, ymax]) { s("line", { x1: marco[0], x2: marco[1], y1: y(v), y2: y(v) }, gr); s("text", { x: marco[0] - 6, y: y(v) + 4, "text-anchor": "end", texto: coma(v, 2) }, ge); }
  FASES.forEach((k, j) => s("text", { x: x(j), y: marco[3] + 16, "text-anchor": "middle", texto: k }, ge));
  s("text", { class: "titulo-eje", x: 12, y: 87, "text-anchor": "middle", transform: "rotate(-90 12 87)", texto: "brillo medio" }, svg);
  for (const q of series) {
    s("polyline", { points: q.v.map((v, j) => `${x(j)},${y(v)}`).join(" "), style: `fill:none;stroke:${q.color};stroke-width:2.5` }, svg);
    q.v.forEach((v, j) => s("circle", { cx: x(j), cy: y(v), r: 5, style: `fill:${q.color};stroke:var(--panel);stroke-width:2`,
                                       "data-tip": `<b>${FASES[j]}</b>, ${q.nombre}<br>brillo medio ${coma(v, 3)}` }, svg));
  }
  const [pre, early, late] = series[0].v;
  const realce = (early - pre) / Math.max(pre, 1e-6), cambio = (late - early) / Math.max(early, 1e-6);
  const forma = cambio > 0.05 ? "sigue subiendo" : cambio < -0.05 ? "lavado: baja tras el pico" : "se mantiene";
  const porc = (v) => { const r = Math.round(100 * v); return `${r > 0 ? "+" : r < 0 ? "−" : ""}${Math.abs(r)} %`; };
  $("l-cinetica").innerHTML = series.map((q) => `<span><i class="linea" style="--c:${q.color}"></i>${q.nombre}</span>`).join("") +
    `<p>En la zona de interés, el brillo sube un <b>${porc(realce)}</b> de PRE a EARLY y cambia un <b>${porc(cambio)}</b> de EARLY a LATE ` +
    `(${forma}). Es descriptivo, no una valoración clínica.</p>`;
}

// ------------------------------------------------------------------------------------------------ vista Predecir
function actualizarBotonPredecir() {
  $("predecir").disabled = !(estado.listo && estado.fases) || estado.procesando;
}

// Las comprobaciones de cada fase: si todas pasan, una sola línea; si alguna falla, se ven todas y cuál falla
function ponerChips(fase, comprobaciones) {
  const caja = document.querySelector(`[data-chips="${fase}"]`);
  if (comprobaciones.length && comprobaciones.every((c) => c.ok)) {
    const visibles = comprobaciones.filter((c) => c.t !== "valores finitos" && !c.t.endsWith(" MB")).map((c) => c.t);   // el resto, en el título
    caja.innerHTML = `<span class="chip bien" title="${escapar(comprobaciones.map((c) => c.t).join(", "))}"><i>✓</i>${escapar(visibles.join(", "))}</span>`;
    return;
  }
  caja.innerHTML = comprobaciones.map((c) => `<span class="chip ${c.ok ? "bien" : "mal"}"><i>${c.ok ? "✓" : "✗"}</i>${escapar(c.t)}</span>`).join("");
}

function cineticaVacia() {
  const svg = $("g-cinetica"); vaciar(svg);
  const ge = s("g", { class: "eje" }, svg), x = (k) => 76 + k * 122;
  s("line", { x1: 46, x2: 350, y1: 160, y2: 160 }, ge);
  FASES.forEach((k, j) => s("text", { x: x(j), y: 176, "text-anchor": "middle", texto: k }, ge));
  s("polyline", { points: `${x(0)},140 ${x(1)},70 ${x(2)},84`, style: "fill:none;stroke:var(--borde-fuerte);stroke-width:2;stroke-dasharray:5 5" }, svg);
  s("text", { x: 198, y: 30, "text-anchor": "middle", texto: "Al cargar una muestra se ve cómo cambia" }, svg);
  s("text", { x: 198, y: 46, "text-anchor": "middle", texto: "el brillo de la zona de interés" }, svg);
  $("l-cinetica").innerHTML = "";
}

// La muestra esta lista cuando las tres fases se han leido bien: entonces se ve en el visor y en la curva de realce
function prepararMuestra() {
  const ok = FASES.every((k) => estado.leidas[k]);
  estado.fases = ok ? { PRE: estado.leidas.PRE, EARLY: estado.leidas.EARLY, LATE: estado.leidas.LATE } : null;
  if (estado.animando) pararSecuencia();
  $("visor-vacio").hidden = ok; $("play").disabled = !ok;
  resultadoInicial();
  $("r-muestra").textContent = ok ? `${nombreMuestra()}: lista para predecir` : "Sin muestra cargada";
  avisoOrden(ok ? estado.fases : null);
  if (ok) { mostrarFase(estado.faseVisor); graficaCinetica(estado.fases); }
  else {
    const lienzo = $("visor"); lienzo.getContext("2d").clearRect(0, 0, lienzo.width, lienzo.height);
    quitarAnotaciones(); $("ov-tres").hidden = true; $("cortina").hidden = true; $("leyenda-visor").innerHTML = ""; cineticaVacia();
  }
  actualizarBotonPredecir();
}

function ponerFichero(fase, buffer, nombre) {
  estado.buffers[fase] = buffer; estado.nombres[fase] = nombre; estado.leidas[fase] = null;
  estado.ejemploActivo = null; estado.loteActivo = null;
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

// Coloca en las ranuras unas fases ya leidas (ejemplos de la maqueta, muestras del lote)
function cargarFasesDirectas(fases, nombres, chips, indice = null) {
  for (const k of FASES) {
    estado.buffers[k] = null; estado.nombres[k] = nombres[k]; estado.leidas[k] = fases[k];
    const ranura = document.querySelector(`.ranura[data-fase="${k}"]`);
    ranura.classList.remove("mal"); ranura.classList.add("cargada");
    document.querySelector(`[data-estado="${k}"]`).textContent = nombres[k];
    ponerChips(k, chips);
    miniatura(document.querySelector(`[data-mini="${k}"]`), fases[k]);
  }
  document.querySelectorAll('.ranura input[type="file"]').forEach((i) => { i.value = ""; });
  estado.ejemploActivo = indice;
  prepararMuestra();
}

function mostrarErrores(lista) {
  const caja = $("errores");
  caja.hidden = !lista.length;
  caja.innerHTML = lista.length ? `<ul>${lista.map((e) => `<li>${escapar(e)}</li>`).join("")}</ul>` : "";
}

// ---- el proceso, paso a paso, mientras se predice ----
// Los cinco pasos del cálculo: en gris mientras no hay predicción. Al predecir, cada paso despliega lo que pasa con
// esta muestra: las tres fases, el recorte, la probabilidad de cada red, la calibración y la decisión frente al umbral.
function procesoReposo() {
  $("proceso").hidden = false;
  document.querySelectorAll("#proceso [data-paso]").forEach((li) => li.classList.remove("activo", "hecho"));
  $("proceso-decision").textContent = `Con el umbral ${coma(D.modelo.umbral, 3)}`;
  ["pv-lectura", "pv-calibracion"].forEach((id) => { $(id).innerHTML = ""; });
  vaciar($("pv-redes")); vaciar($("pv-decision")); $("pv-cuenta").textContent = "";
}
function procesoIniciar() {
  $("resultado").hidden = true;
  const aviso = $("resultado-vacio");      // la frase se queda en su sitio (nada salta) y cambia a "calculando"
  aviso.hidden = false; aviso.textContent = "Calculando con las quince redes…"; aviso.classList.add("calculando");
  procesoReposo();
  $("proceso").classList.remove("saliendo");
}
function fundirFoco(hasta, ms) {
  const desde = estado.foco;
  return new Promise((listo) => animar(ms, (k) => {
    estado.foco = desde + (hasta - desde) * k;
    if (estado.fases) dibujarVisor(null, estado.faseVisor);
    if (k >= 1) listo();
  }));
}
function procesoMarcar(i, clase) {
  const li = document.querySelector(`#proceso [data-paso="${i}"]`);
  li.classList.remove("activo", "hecho"); li.classList.add(clase);
}

// Al pulsar Predecir: el informe a la vista (a la derecha en pantallas grandes, debajo del visor en las demás)
function llevarAlInforme() {
  const panel = document.querySelector(".panel-informe"), suave = SIN_MOVIMIENTO ? "auto" : "smooth";
  const r = panel.getBoundingClientRect();
  if (r.top < 56 || r.top > window.innerHeight * 0.45) panel.scrollIntoView({ behavior: suave, block: "start" });
  if (panel.scrollHeight > panel.clientHeight) panel.scrollTo({ top: 0, behavior: suave });
}

function pvLectura(fases) {
  $("pv-lectura").innerHTML = "";
  for (const k of FASES) {
    const fig = document.createElement("figure"), c = document.createElement("canvas");
    c.width = c.height = LADO; pintarGris(c.getContext("2d"), fases[k]);
    fig.append(c); fig.insertAdjacentHTML("beforeend", `<figcaption>${k}</figcaption>`);
    $("pv-lectura").append(fig);
  }
}

// La miniatura de EARLY hace zoom hasta el cuadrado central de 128 × 128
function pvRecorte(fases, ms) {
  const fuente = document.createElement("canvas"); fuente.width = fuente.height = LADO;
  pintarGris(fuente.getContext("2d"), fases.EARLY);
  const ctx = $("pv-recorte").getContext("2d");
  return new Promise((listo) => {
    animar(ms, (k) => {
      const lado = LADO - k * RECORTE, o = k * INI;
      ctx.drawImage(fuente, o, o, lado, lado, 0, 0, LADO, LADO);
      const a = (INI - o) * LADO / lado, b = RECORTE * LADO / lado;      // el cuadrado, en coordenadas del lienzo
      ctx.save(); ctx.strokeStyle = ANOTACION; ctx.lineWidth = 3; ctx.setLineDash([10, 8]);
      ctx.strokeRect(a + 1.5, a + 1.5, b - 3, b - 3); ctx.restore();
      if (k >= 1) listo();
    });
  });
}

// Una barra por red, agrupadas por fold (tres semillas en cada grupo), que crecen hasta su probabilidad al terminar
// cada red; la línea amarilla es la media de las que van
function pvRedesIniciar() {
  const svg = $("pv-redes"); vaciar(svg);
  const y = escala(0, 0.6, 74, 6), base = y(0), ancho = 12, hueco = 3, grupo = 3 * ancho + 2 * hueco, sep = (292 - 34 - 5 * grupo) / 4;
  const xBarra = (i) => 34 + (i % 5) * (grupo + sep) + Math.floor(i / 5) * (ancho + hueco);
  const ge = s("g", { class: "eje" }, svg), gr = s("g", { class: "rejilla" }, svg);
  for (const v of [0, 0.3, 0.6]) {
    s("line", { x1: 30, x2: 292, y1: y(v), y2: y(v) }, gr);
    s("text", { x: 26, y: y(v) + 3, "text-anchor": "end", texto: coma(v, 1) }, ge);
  }
  for (let k = 0; k < 5; k++) s("text", { x: 34 + k * (grupo + sep) + grupo / 2, y: 87, "text-anchor": "middle", texto: `fold ${k}` }, ge);
  const media = s("line", { x1: 30, x2: 292, y1: base, y2: base, style: "stroke:var(--anotacion-texto);stroke-width:2;opacity:0" }, svg);
  const etiqueta = s("text", { x: 292, y: base - 4, "text-anchor": "end", style: "fill:var(--anotacion-texto);font-weight:700;opacity:0", texto: "" }, svg);
  const probs = [];
  $("pv-cuenta").textContent = "0 de 15 redes";
  return (i, p) => {
    probs.push(p);
    const barra = s("rect", { x: xBarra(i), width: ancho, y: base, height: 0, rx: 1, style: "fill:var(--texto)" }, svg);
    barra.appendChild(s("title", { texto: `fold ${i % 5}, semilla ${SEMILLAS[Math.floor(i / 5)]}: ${coma(p, 3)}` }));
    const alto = base - y(Math.min(p, 0.6));
    animar(320, (k) => { barra.setAttribute("y", base - alto * k); barra.setAttribute("height", alto * k); });
    const m = probs.reduce((a, b) => a + b, 0) / probs.length;
    media.setAttribute("y1", y(m)); media.setAttribute("y2", y(m)); media.style.opacity = 1;
    etiqueta.setAttribute("y", y(m) - 4); etiqueta.textContent = `media ${coma(m, 3)}`; etiqueta.style.opacity = 1;
    svg.appendChild(media); svg.appendChild(etiqueta);       // la media, siempre por encima de las barras
    $("pv-cuenta").textContent = `${probs.length} de 15 redes`;
  };
}

function pvCalibracion(r, ms) {
  $("pv-calibracion").innerHTML = `<span>media de las redes</span><b>${coma(r.media, 3)}</b><i>→</i><span>calibrada</span><b id="pv-calibrada">${coma(r.media, 3)}</b>`;
  return new Promise((listo) => animar(ms, (k) => {
    $("pv-calibrada").textContent = coma(r.media + (r.probabilidad - r.media) * k, 3);
    if (k >= 1) listo();
  }));
}

// Barra de 0 a 1: la marca se desliza hasta la probabilidad calibrada, frente al umbral
function pvDecision(r, ms) {
  const svg = $("pv-decision"); vaciar(svg);
  const u = D.modelo.umbral, x = escala(0, 1, 8, 292), yb = 22;
  s("line", { x1: 8, x2: x(u), y1: yb, y2: yb, style: "stroke:var(--nopcr);stroke-width:6;stroke-linecap:round" }, svg);
  s("line", { x1: x(u), x2: 292, y1: yb, y2: yb, style: "stroke:var(--pcr);stroke-width:6;stroke-linecap:round" }, svg);
  s("line", { x1: x(u), x2: x(u), y1: yb - 9, y2: yb + 9, style: "stroke:var(--texto-2);stroke-width:1.5" }, svg);
  s("text", { x: x(u), y: yb + 22, "text-anchor": "middle", texto: `umbral ${coma(u, 3)}` }, svg);
  s("text", { x: 8, y: yb + 22, texto: "0" }, svg); s("text", { x: 292, y: yb + 22, "text-anchor": "end", texto: "1" }, svg);
  const g = s("g", {}, svg);
  s("circle", { cx: 0, cy: yb, r: 7, style: "fill:var(--texto);stroke:var(--panel);stroke-width:2" }, g);
  return new Promise((listo) => animar(ms, (k) => {
    g.setAttribute("transform", `translate(${x(r.probabilidad * k)} 0)`);
    if (k >= 1) listo();
  }));
}

async function predecir() {
  mostrarErrores([]);
  const fases = estado.fases;
  if (!fases || estado.procesando) return;
  estado.procesando = true;
  const boton = $("predecir");
  boton.classList.add("calculando"); boton.disabled = true;
  const lento = !SIN_MOVIMIENTO, t = (ms) => (lento ? ms : 0);
  try {
    procesoIniciar();
    $("r-muestra").textContent = `${nombreMuestra()}: calculando…`;
    llevarAlInforme();
    procesoMarcar(0, "activo"); pvLectura(fases); await esperar(t(650)); procesoMarcar(0, "hecho");
    procesoMarcar(1, "activo"); fundirFoco(1, t(500));
    await pvRecorte(fases, t(900)); await esperar(t(200)); procesoMarcar(1, "hecho");
    procesoMarcar(2, "activo");
    const r = await predecirCorte(fases, { alRed: pvRedesIniciar(), pausa: t(75) });
    await esperar(t(250)); procesoMarcar(2, "hecho"); fundirFoco(0, t(500));
    const brillo = predecirBrillo(fases, r);      // mientras se anima la calibración
    procesoMarcar(3, "activo"); await pvCalibracion(r, t(800)); await esperar(t(200)); procesoMarcar(3, "hecho");
    procesoMarcar(4, "activo");
    $("proceso-decision").textContent = `${r.clase === "pCR" ? "pCR = 1" : "pCR = 0"}: ${coma(r.probabilidad, 3)} frente al umbral ${coma(D.modelo.umbral, 3)}`;
    await pvDecision(r, t(800)); await esperar(t(650)); procesoMarcar(4, "hecho");
    $("proceso").classList.add("saliendo"); await esperar(t(300));
    r.brillo = await brillo;
    mostrarResultado(r, fases);
    const res = $("resultado"); res.classList.remove("entrando"); void res.offsetWidth; res.classList.add("entrando");
    anadirHistorial(r, fases);
    llevarAlInforme();
  } catch (e) {
    console.error(e); estado.foco = 0; resultadoInicial();
    mostrarErrores([`No se ha podido calcular la predicción (${e.message}).`]);
  } finally {
    estado.procesando = false;
    if (estado.fases) dibujarVisor(null, estado.faseVisor);
    boton.classList.remove("calculando"); actualizarBotonPredecir();
  }
}

// Antes de la primera prediccion: el contexto (sin marca) y una nota en la seccion de las 15 redes
function resultadoInicial() {
  $("resultado").hidden = true; procesoReposo(); $("proceso").classList.remove("saliendo");
  const aviso = $("resultado-vacio");
  aviso.hidden = false; aviso.classList.remove("calculando"); aviso.textContent = "Carga las tres fases de un corte, o elige un ejemplo, y pulsa Predecir.";
  $("r-hora").textContent = "";
  histograma($("g-contexto"), 360, 170, D.modelo.umbral);
  $("r-contexto").innerHTML = `Así se reparten las probabilidades de las 1.097 pacientes de validación; el ${pct(PREVALENCIA)} tuvo pCR. ` +
                              "Al predecir, una línea marcará dónde cae la muestra.";
  const svg = $("g-redes"); vaciar(svg);
  const ge = s("g", { class: "eje" }, svg);
  for (let k = 0; k < 5; k++) s("text", { x: 44, y: 20 + k * 20, "text-anchor": "end", texto: `fold ${k}` }, ge);
  s("text", { x: 200, y: 62, "text-anchor": "middle", texto: "Al predecir aparece la probabilidad" }, svg);
  s("text", { x: 200, y: 80, "text-anchor": "middle", texto: "que da cada una de las quince redes" }, svg);
  $("r-tiempo").textContent = "—";
  $("r-redes").textContent = "";
}

// Ejemplos (pacientes de entrenamiento): se conoce su pCR real. 12 de las 15 redes vieron a la paciente al entrenar;
// las 3 de su fold no la vieron (para ellas era validación). Su media, calibrada con Platt igual que en la validación,
// es la comprobación honesta.
function comprobacionReal(r) {
  if (DEMO || estado.ejemploActivo === null || estado.loteActivo !== null) return null;
  const e = EJEMPLOS[estado.ejemploActivo];
  if (!e || r.probs.length !== 15) return null;
  const tres = r.probs.filter((_, j) => j % 5 === e.fold);
  const m = limitar(tres.reduce((a, b) => a + b, 0) / tres.length, 1e-6, 1 - 1e-6);
  const p3 = sigmoide(D.modelo.calibracion.a * Math.log(m / (1 - m)) + D.modelo.calibracion.b);
  const real = e.pCR === 1, c15 = r.clase === "pCR", c3 = p3 >= D.modelo.umbral;
  return { real, acierta15: c15 === real, p3, acierta3: c3 === real, clase3: c3, fold: e.fold };
}

function mostrarReal(r) {
  const c = comprobacionReal(r);
  $("r-real").hidden = !c;
  r.real = c;
  if (!c) return;
  const v = $("r-veredicto");
  v.className = `insignia-veredicto ${c.acierta15 ? "acierta" : "falla"}`;
  v.textContent = c.acierta15 ? "✓ Acierta" : "✗ Falla";
  $("r-real-texto").textContent = `pCR real = ${c.real ? 1 : 0}: ${c.real ? "tuvo respuesta completa" : "no tuvo respuesta completa"}.`;
  $("r-real-nota").innerHTML = `Es una paciente de entrenamiento: 12 de las 15 redes la vieron al entrenar. Las 3 que no la vieron (fold ${c.fold}) ` +
    `dan <b>${cifraUmbral(c.p3)}</b> calibrada, es decir, <b>pCR = ${c.clase3 ? 1 : 0}</b>: ` +
    `${c.acierta3 === c.acierta15 ? "también " : "en cambio, "}${c.acierta3 ? "aciertan" : "fallan"}. ` +
    "Así se midió el modelo en validación.";
}

// Una probabilidad con 3 decimales, o con 4 si está tan cerca del umbral que con 3 parecería igual
const cifraUmbral = (p) => coma(p, Math.abs(p - D.modelo.umbral) < 0.0005 ? 4 : 3);

// Distancia al umbral y si la clase aguanta un 10 % más o menos de brillo
function mostrarAjuste(r, u) {
  const el = $("r-ajuste"), b = r.brillo;
  el.hidden = !b;
  r.ajustada = null;
  if (!b) return;
  const valores = [r.probabilidad, b.oscura, b.clara], lo = Math.min(...valores), hi = Math.max(...valores);
  const ajustada = valores.some((v) => (v >= u) !== (r.probabilidad >= u)), d = Math.abs(r.probabilidad - u), distancia = coma(d, d < 0.0005 ? 4 : 3);
  r.ajustada = ajustada;
  el.className = `dictamen-ajuste${ajustada ? " ajustada" : ""}`;
  el.innerHTML = ajustada
    ? `<span class="insignia-ajuste">Decisión ajustada</span>Está a ${distancia} del umbral: con un 10 % más o menos de brillo, la probabilidad iría de ${coma(lo, 3)} a ${coma(hi, 3)} y cambiaría de clase.`
    : `Está a ${distancia} del umbral. Con un 10 % más o menos de brillo iría de ${coma(lo, 3)} a ${coma(hi, 3)}: la clase se mantiene.`;
}

function mostrarResultado(r, fases) {
  const u = D.modelo.umbral, pac = D.validacion.pacientes, pcr = r.clase === "pCR";
  $("resultado-vacio").hidden = true; $("proceso").hidden = true; $("resultado").hidden = false;
  $("r-muestra").textContent = `${nombreMuestra()}, un corte`;
  $("r-hora").textContent = new Date().toLocaleTimeString("es-ES");
  $("r-dictamen").className = `dictamen ${pcr ? "pcr" : "nopcr"}`;
  contarHasta($("r-prob"), r.probabilidad);
  $("r-clase").textContent = pcr ? "pCR = 1" : "pCR = 0";
  $("r-clase-texto").textContent = pcr ? "respuesta completa probable" : "respuesta completa poco probable";
  $("r-frase").textContent = `La probabilidad calibrada ${pcr ? "supera el" : "queda por debajo del"} umbral de ${coma(u, 3)}.`;
  medidor(r.probabilidad, u);
  mostrarAjuste(r, u);
  mostrarReal(r);
  const menores = pac.filter(([p]) => p < r.probabilidad).length;
  $("r-percentil").textContent = `Arriba, las 1.097 pacientes de validación. Esta muestra supera la probabilidad del ${pct(menores / pac.length)} de ellas.`;
  const cerca = pac.filter(([p]) => Math.abs(p - r.probabilidad) <= 0.02);
  const hallazgos = [
    `Las quince redes dan entre <b>${coma(Math.min(...r.probs), 3)}</b> y <b>${coma(Math.max(...r.probs), 3)}</b>; su media es <b>${coma(r.media, 3)}</b>.`,
    `La calibración de Platt lleva esa media a <b>${coma(r.probabilidad, 3)}</b>, la probabilidad que se compara con el umbral.`,
    cerca.length >= 10
      ? `De las ${cerca.length} pacientes de validación con una probabilidad parecida (±0,02), el <b>${pct(cerca.filter(([, y]) => y).length / cerca.length)}</b> tuvo pCR, frente al ${pct(PREVALENCIA)} del total.`
      : "Hay pocas pacientes de validación con una probabilidad tan extrema como esta.",
  ];
  $("r-hallazgos").innerHTML = hallazgos.map((h) => `<li>${h}</li>`).join("");
  $("r-calidad").innerHTML = controlCalidad(fases).map((c) => `<li class="${c.ok ? "" : "mal"}">${escapar(c.texto)}</li>`).join("");
  histograma($("g-contexto"), 360, 170, u, r.probabilidad);
  $("r-contexto").innerHTML = (cerca.length >= 10
    ? `De las ${cerca.length} pacientes de validación con una probabilidad parecida (±0,02), el ${pct(cerca.filter(([, y]) => y).length / cerca.length)} tuvo pCR; en total fue el ${pct(PREVALENCIA)}. `
    : "Hay pocas pacientes de validación con una probabilidad tan extrema. ") + `Su probabilidad supera a la del ${pct(menores / pac.length)} de ellas.`;
  graficaRedes(r.probs);
  $("r-redes").innerHTML = `Las quince redes dan entre ${coma(Math.min(...r.probs))} y ${coma(Math.max(...r.probs))}. Su media, ` +
    `${coma(r.media)}, pasa a <b>${coma(r.probabilidad)}</b> con la calibración de Platt.`;
  $("r-tiempo").textContent = DEMO ? "— (maqueta)" : `${coma(r.segundos * 1000, 0)} ms para las quince redes`;
}

function anadirHistorial(r, fases, nombreDado = null) {
  const n = estado.historial.length + 1, hora = new Date().toLocaleTimeString("es-ES");
  const nombre = nombreDado || nombreMuestra();
  const fila = { n, hora, nombre, probabilidad: r.probabilidad, clase: r.clase, ms: r.segundos * 1000, checksum: DEMO ? "maqueta" : estado.checksumWeb,
                 real: r.real ? (r.real.real ? 1 : 0) : null, acierta: r.real ? r.real.acierta15 : null,
                 ajustada: typeof r.ajustada === "boolean" ? r.ajustada : null };
  estado.historial.push(fila);
  const tbody = $("historial");
  if (n === 1) tbody.innerHTML = "";
  const tr = document.createElement("tr");
  const td = (texto, clase) => { const c = document.createElement("td"); if (clase) c.className = clase; c.innerHTML = texto; return c; };
  tr.append(td(String(n)), td(hora));
  const celda = td(""); const c = document.createElement("canvas"); c.width = c.height = LADO; pintarGris(c.getContext("2d"), fases.EARLY); celda.appendChild(c); tr.append(celda);
  tr.append(td(escapar(nombre)), td(coma(r.probabilidad, 3), "num"),
            td(`<span class="punto-clase" style="--c:var(${r.clase === "pCR" ? "--pcr" : "--nopcr"})">${r.clase === "pCR" ? "pCR = 1" : "pCR = 0"}</span>`),
            td(fila.real === null ? "—" : `${fila.real} <span class="${fila.acierta ? "acierta" : "falla"}">${fila.acierta ? "✓ acierta" : "✗ falla"}</span>`),
            td(DEMO ? "—" : `${coma(fila.ms, 0)} ms`, "num"), td(fila.checksum, "mono"));
  tbody.prepend(tr);
  $("descargar").disabled = false; $("vaciar").disabled = false;
}

function descargarHistorial() {
  const filas = [["n", "hora", "muestra", "probabilidad", "clase", "umbral", "decision_ajustada", "pcr_real", "acierta", "tiempo_ms", "checksum"]];
  for (const h of estado.historial) filas.push([h.n, h.hora, h.nombre, coma(h.probabilidad, 6), h.clase === "pCR" ? "pCR=1" : "pCR=0", coma(D.modelo.umbral, 6),
    h.ajustada === null ? "" : (h.ajustada ? "si" : "no"), h.real === null ? "" : h.real, h.acierta === null ? "" : (h.acierta ? "si" : "no"), DEMO ? "" : coma(h.ms, 0), h.checksum]);
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
  document.querySelectorAll(".lote-item").forEach((b) => b.classList.remove("activa"));
}

// ---- varias muestras: la fase se reconoce por el nombre del fichero (…_PRE.png, …_EARLY.png, …_LATE.png) ----
function faseDelNombre(nombre) {
  const base = nombre.replace(/\.[^.]+$/, "");
  const m = base.match(/(^|[_\-\s.])(PRE|EARLY|LATE)(?=$|[_\-\s.])/i);
  if (!m) return null;
  const clave = (base.slice(0, m.index) + base.slice(m.index + m[0].length)).replace(/^[_\-\s.]+|[_\-\s.]+$/g, "") || "muestra";
  return { fase: m[2].toUpperCase(), clave };
}

function agrupar(ficheros) {
  const grupos = new Map(), errores = [];
  for (const f of ficheros) {
    const d = faseDelNombre(f.name);
    if (!d) { errores.push(`${f.name}: el nombre no indica la fase. Debe acabar en _PRE, _EARLY o _LATE.`); continue; }
    if (!grupos.has(d.clave)) grupos.set(d.clave, {});
    const g = grupos.get(d.clave);
    if (g[d.fase]) errores.push(`${d.clave}: hay dos ficheros ${d.fase}; se usa el último.`);
    g[d.fase] = f;
  }
  return { grupos: [...grupos].sort((a, b) => a[0].localeCompare(b[0], "es", { numeric: true })), errores };
}

// Varios ficheros soltados de golpe: si son las tres fases de una muestra van a las ranuras; si son mas, al lote
async function repartirFicheros(lista) {
  const ficheros = [...lista];
  const { grupos, errores } = agrupar(ficheros);
  if (grupos.length === 1 && !errores.length && FASES.every((k) => grupos[0][1][k])) {
    const g = grupos[0][1];
    for (const k of FASES) ponerFichero(k, await g[k].arrayBuffer(), g[k].name);
    return;
  }
  await procesarLote(ficheros);
}

async function procesarLote(ficheros) {
  if (!estado.listo) { mostrarErrores(["El modelo aún se está cargando. Espera a que la barra de abajo ponga «Modelo listo»."]); return; }
  const { grupos, errores } = agrupar(ficheros);
  const tarjeta = $("lote"), rejilla = $("lote-rejilla");
  tarjeta.hidden = false; rejilla.innerHTML = ""; estado.lote = []; estado.loteActivo = null;
  tarjeta.scrollIntoView({ behavior: SIN_MOVIMIENTO ? "auto" : "smooth", block: "nearest" });
  let hechas = 0;
  for (const [clave, g] of grupos) {
    $("lote-resumen").textContent = `Procesando ${++hechas} de ${grupos.length}…`;
    const faltan = FASES.filter((k) => !g[k]);
    if (faltan.length) { errores.push(`${clave}: falta ${faltan.join(" y ")}.`); continue; }
    try {
      const fases = {};
      for (const k of FASES) fases[k] = leerFase(await g[k].arrayBuffer(), `${clave}, ${k}`).x;
      const r = await predecirCorte(fases);
      const item = { clave, fases, r, avisos: avisos(fases), nombres: Object.fromEntries(FASES.map((k) => [k, g[k].name])) };
      estado.lote.push(item);
      anadirHistorial(r, fases, clave);
      pintarItemLote(item, estado.lote.length - 1);
    } catch (e) { errores.push(e instanceof EntradaNoValida ? e.message : `${clave}: ${e.message}`); }
  }
  const n = estado.lote.length, unos = estado.lote.filter((q) => q.r.clase === "pCR").length;
  $("lote-resumen").textContent = n ? `${n} muestra${n === 1 ? "" : "s"}: ${unos} con pCR = 1 y ${n - unos} con pCR = 0` : "Ninguna muestra completa";
  const caja = $("lote-errores");
  caja.hidden = !errores.length;
  caja.innerHTML = errores.length ? `<ul>${errores.map((e) => `<li>${escapar(e)}</li>`).join("")}</ul>` : "";
  if (n) abrirDelLote(0, false);
}

function pintarItemLote(item, i) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "lote-item";
  const c = document.createElement("canvas"); c.width = c.height = LADO;
  const ctx = c.getContext("2d"); pintarGris(ctx, item.fases.EARLY);
  const pcr = item.r.clase === "pCR";
  b.append(c);
  b.insertAdjacentHTML("beforeend", `<span class="lote-nombre">${escapar(item.clave)}</span>
    <span class="lote-dato"><b>${coma(item.r.probabilidad, 3)}</b><span class="punto-clase" style="--c:var(${pcr ? "--pcr" : "--nopcr"})">${pcr ? "pCR = 1" : "pCR = 0"}</span></span>
    ${item.avisos.length ? `<span class="lote-aviso" title="${escapar(item.avisos.join(" "))}">${item.avisos.length} aviso${item.avisos.length > 1 ? "s" : ""}</span>` : ""}`);
  b.addEventListener("click", () => abrirDelLote(i, true));
  $("lote-rejilla").append(b);
}

function abrirDelLote(i, desplazar) {
  const item = estado.lote[i];
  if (!item) return;
  document.querySelectorAll(".lote-item").forEach((b, j) => b.classList.toggle("activa", j === i));
  cargarFasesDirectas(item.fases, item.nombres, [{ t: "PNG", ok: true }, { t: "256 × 256", ok: true }]);
  estado.loteActivo = i;
  mostrarResultado(item.r, item.fases);
  dibujarVisor(null, estado.faseVisor);
  if (desplazar) document.querySelector(".puesto").scrollIntoView({ behavior: SIN_MOVIMIENTO ? "auto" : "smooth", block: "start" });
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
    f.PRE[i] = limitar(pre + ruido(), 0, 1); f.EARLY[i] = limitar(early + ruido(), 0, 1); f.LATE[i] = limitar(late + ruido(), 0, 1);
  }
  return f;
}

async function bajar(ruta) { const r = await fetch(ruta); if (!r.ok) throw new Error(`${ruta}: ${r.status}`); return r.arrayBuffer(); }

const COHORTE = { spy1: "I-SPY1", spy2: "I-SPY2", duke: "Duke" };
function montarGaleria() {
  const g = $("galeria"); g.innerHTML = "";
  if (DEMO) {
    $("ejemplos-nota").textContent = "Fantasmas sintéticos: en la maqueta no hay imágenes reales.";
    for (let i = 0; i < 6; i++) {
      const f = fantasma(1000 + i * 77), b = document.createElement("button");
      b.className = "ejemplo"; b.type = "button";
      const c = document.createElement("canvas"); c.width = c.height = LADO; pintarGris(c.getContext("2d"), f.EARLY);
      b.append(c); b.insertAdjacentHTML("beforeend", `<span>Fantasma ${i + 1}</span>`); b.title = `Fantasma ${i + 1}, sintético`;
      const nombres = Object.fromEntries(FASES.map((k) => [k, `Fantasma ${i + 1} (${k})`]));
      b.addEventListener("click", () => { estado.loteActivo = null; cargarFasesDirectas(f, nombres, [{ t: "fantasma sintético", ok: true }, { t: "256 × 256", ok: true }], i); });
      g.append(b);
    }
    return;
  }
  if (!EJEMPLOS.length) { g.innerHTML = '<p class="ayuda" style="grid-column:1/-1">Los ejemplos se generan con galeria_web.py.</p>'; return; }
  $("ejemplos-nota").textContent = "Pacientes de entrenamiento en las que el tumor se ve bien, elegidas sin mirar lo que predice el modelo. La mayoría de las redes ya las vieron.";
  EJEMPLOS.forEach((e, i) => {
    const b = document.createElement("button"); b.className = "ejemplo"; b.type = "button";
    b.innerHTML = `<img src="${e.rutas.EARLY}" alt="" loading="lazy"><span>pCR = ${e.pCR}</span>`;
    b.title = `${e.sample_id}${e.cohorte ? ` (${COHORTE[e.cohorte] || e.cohorte})` : ""}: pCR real = ${e.pCR}`;
    b.setAttribute("aria-label", `Ejemplo ${e.sample_id}, pCR real ${e.pCR}`);
    b.addEventListener("click", async () => {
      mostrarErrores([]);
      try {
        const buffers = await Promise.all(FASES.map((k) => bajar(e.rutas[k])));
        document.querySelectorAll('.ranura input[type="file"]').forEach((inp) => { inp.value = ""; });
        FASES.forEach((k, j) => ponerFichero(k, buffers[j], e.rutas[k].split("/").pop()));
        estado.ejemploActivo = i;
        if (estado.fases) dibujarVisor(null, estado.faseVisor);
      } catch (err) { mostrarErrores([`No se ha podido cargar el ejemplo (${err.message}).`]); }
    });
    g.append(b);
  });
}

// ------------------------------------------------------------------------------------------------ portada
function curvaPortada(serie) {   // brillo de la zona de interes en las tres fases, con un punto que sigue a la animacion
  const svg = $("portada-curva"); vaciar(svg);
  const marco = [44, 424, 14, 86], ymax = Math.max(...serie) * 1.18 || 1;
  const x = (k) => marco[0] + k * (marco[1] - marco[0]) / 2, y = escala(0, ymax, marco[3], marco[2]);
  const ge = s("g", { class: "eje" }, svg);
  s("line", { x1: marco[0] - 20, x2: marco[1] + 10, y1: marco[3], y2: marco[3] }, ge);
  FASES.forEach((k, j) => s("text", { x: x(j), y: marco[3] + 20, "text-anchor": "middle", texto: k }, ge));
  s("polyline", { points: serie.map((v, j) => `${x(j)},${y(v)}`).join(" "), style: "fill:none;stroke:var(--anotacion-texto);stroke-width:2" }, svg);
  serie.forEach((v, j) => s("circle", { cx: x(j), cy: y(v), r: 4, style: "fill:var(--portada-fondo);stroke:var(--anotacion-texto);stroke-width:2" }, svg));
  const punto = s("circle", { cx: x(0), cy: y(serie[0]), r: 6.5, style: "fill:var(--anotacion-texto)" }, svg);
  return (pos) => {
    const k = Math.min(Math.floor(pos), 1), t = limitar(pos - k, 0, 1);
    punto.setAttribute("cx", x(k) + t * (x(k + 1) - x(k)));
    punto.setAttribute("cy", y(serie[k] + t * (serie[k + 1] - serie[k])));
  };
}

function animarPortada(fases, nombre) {
  const canvas = $("portada-canvas"), ctx = canvas.getContext("2d");
  const mover = curvaPortada(FASES.map((k) => mediaDe(fases[k], MASCARA_CENTRO)));
  const indice = { PRE: 0, EARLY: 1, LATE: 2 };
  $("pov-br").innerHTML = `${escapar(nombre)}<br>Un corte`;
  const cuadro = (datos, etiqueta) => {
    pintarGris(ctx, datos);
    ctx.save(); ctx.strokeStyle = ANOTACION; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2; ctx.strokeRect(INI + 0.6, INI + 0.6, RECORTE - 1.2, RECORTE - 1.2); ctx.restore();
    $("pov-tl").innerHTML = `<b>${etiqueta}</b><br>${DESCRIPCION[etiqueta] || "Llegada del contraste"}`;
  };
  if (SIN_MOVIMIENTO) { cuadro(fases.EARLY, "EARLY"); mover(1); return; }
  // la unica animacion que no pide nadie: PRE -> EARLY -> LATE y vuelta, con pausas en cada fase
  const tramos = [["PRE", "PRE", 900], ["PRE", "EARLY", 1500], ["EARLY", "EARLY", 1000], ["EARLY", "LATE", 1500], ["LATE", "LATE", 1100], ["LATE", "PRE", 900]];
  const ciclo = tramos.reduce((t, q) => t + q[2], 0), inicio = performance.now();
  const paso = (ahora) => {
    if ($("portada").hidden) { estado.portada = null; return; }
    let t = (ahora - inicio) % ciclo, k = 0;
    while (t > tramos[k][2]) { t -= tramos[k][2]; k++; }
    const [a, b, dur] = tramos[k], u = t / dur, suave = u * u * (3 - 2 * u);
    cuadro(a === b ? fases[a] : mezcla(fases[a], fases[b], suave), a === b ? a : `${a} → ${b}`);
    mover(a === b ? indice[a] : indice[a] + (indice[b] - indice[a]) * suave);
    estado.portada = requestAnimationFrame(paso);
  };
  estado.portada = requestAnimationFrame(paso);
}

async function fasesDePortada() {
  if (DEMO || !EJEMPLOS.length) return { fases: fantasma(1000), nombre: "Fantasma sintético" };
  try {
    const e = EJEMPLOS[0], f = {};
    for (const k of FASES) f[k] = leerFase(await bajar(e.rutas[k]), k).x;
    return { fases: f, nombre: e.sample_id };
  } catch (err) { return { fases: fantasma(1000), nombre: "Fantasma sintético" }; }
}

async function abrirPortada() {
  const p = $("portada"), T = D.comparacion && D.comparacion.test, v = D.modelo.metricas_validacion;
  $("portada-resultado").textContent = T
    ? `En el test, con ${T.pacientes_n} pacientes que el modelo no había visto, separa las que responden de las que no con un AUC de ${coma(T.modelos.recorte.auc, 3)} (${coma(v.auc, 3)} en validación, con 1.097 pacientes).`
    : `En validación, con 1.097 pacientes, separa las que responden de las que no con un AUC de ${coma(v.auc, 3)}.`;
  p.hidden = false;
  requestAnimationFrame(() => p.classList.add("visible"));
  document.body.classList.add("sin-desplazar");
  if (estado.portada) cancelAnimationFrame(estado.portada);
  const { fases, nombre } = await fasesDePortada();
  animarPortada(fases, nombre);
  setTimeout(() => p.focus({ preventScroll: true }), 50);   // el foco entra en la portada sin marcar ningún botón
}

function cerrarPortada(vista) {
  const p = $("portada");
  p.classList.remove("visible");
  document.body.classList.remove("sin-desplazar");
  setTimeout(() => { p.hidden = true; if (estado.portada) { cancelAnimationFrame(estado.portada); estado.portada = null; } }, SIN_MOVIMIENTO ? 0 : 360);
  try { sessionStorage.setItem("portada", "vista"); } catch (e) {}
  if (vista) cambiarVista(vista);
}

// ------------------------------------------------------------------------------------------------ vista Modelo
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

function curvaRocDe(pares) {     // pares [puntuacion, y] -> puntos [1 - especificidad, sensibilidad]
  const orden = [...pares].sort((a, b) => b[0] - a[0]);
  const nPos = orden.filter(([, y]) => y).length, nNeg = orden.length - nPos, puntos = [[0, 0]];
  let vp = 0, fp = 0;
  for (let i = 0; i < orden.length; i++) {
    if (orden[i][1]) vp++; else fp++;
    if (i === orden.length - 1 || orden[i + 1][0] !== orden[i][0]) puntos.push([fp / nNeg, vp / nPos]);
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
  s("circle", { cx: x(1 - m.esp), cy: y(m.sens), r: 7, style: "fill:var(--anotacion-texto);stroke:var(--consola);stroke-width:3",
                "data-tip": `umbral ${coma(u, 3)}<br>sensibilidad ${pct(m.sens)}, especificidad ${pct(m.esp)}` }, svg);
  s("text", { class: "marca-texto", x: x(0.55), y: y(0.12), texto: `AUC ${coma(D.modelo.metricas_validacion.auc, 3)}` }, svg);
}

function matrizHtml(vn, fp, fn, vp) {
  return `<div></div><div class="cab">predicción<br>pCR = 0</div><div class="cab">predicción<br>pCR = 1</div>
    <div class="cab">real<br>pCR = 0</div><div class="celda acierto"><b>${vn}</b>verdaderos negativos</div><div class="celda"><b>${fp}</b>falsos positivos</div>
    <div class="cab">real<br>pCR = 1</div><div class="celda"><b>${fn}</b>falsos negativos</div><div class="celda acierto"><b>${vp}</b>verdaderos positivos</div>`;
}

function actualizarUmbral(u) {
  $("u-valor").textContent = coma(u, 3);
  const m = metricasUmbral(u);
  $("u-matriz").innerHTML = matrizHtml(m.vn, m.fp, m.fn, m.vp);
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
  grupos.forEach(([a, b, n], k) => s("circle", { cx: x(a), cy: y(b), r: 5.5, style: "fill:var(--acento);stroke:var(--consola);stroke-width:2",
    "data-tip": `<b>grupo ${k + 1}</b> (${n} pacientes)<br>predicho ${coma(a, 3)}, real ${coma(b, 3)}` }, svg));
  const brier = mediaDe(D.validacion.pacientes.map(([p, y]) => (p - y) ** 2));
  s("text", { class: "marca-texto", x: x(lim * 0.04), y: y(lim * 0.92), texto: `Brier ${coma(brier, 4)} (sin mirar la imagen: ${coma(D.modelo.metricas_validacion.brier_referencia, 4)})` }, svg);
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
      "data-tip": `<b>época ${e}</b>${e === c.elegida ? " (elegida)" : ""}<br>entrenamiento ${coma(c[a][j], 3)}, validación ${coma(c[b][j], 3)}` }, svg));
  };
  pintar($("g-perdida"), "perdida_train", "perdida_val", c.referencia_val, "pérdida", (v) => coma(v, 3));
  pintar($("g-auc"), "auc_train", "auc_val", 0.5, "AUC por paciente", (v) => coma(v, 2),
         [Math.min(0.4, ...c.auc_train, ...c.auc_val), Math.max(0.8, ...c.auc_train)]);
}

function dibujarArquitectura() {
  const svg = $("g-arquitectura"); vaciar(svg);
  const capas = [
    { n: "Entrada", f: "3 × 256 × 256", c: 3, l: 256, p: 0, d: "PRE, EARLY y LATE" },
    { n: "Recorte", f: "3 × 128 × 128", c: 3, l: 128, p: 0, d: "la zona de interés, en el centro" },
    { n: "Bloque 1", f: "16 × 64 × 64", c: 16, l: 64, p: 464, d: "convolución 3 × 3, BatchNorm, ReLU y pooling" },
    { n: "Bloque 2", f: "32 × 32 × 32", c: 32, l: 32, p: 4672, d: "convolución 3 × 3, BatchNorm, ReLU y pooling" },
    { n: "Bloque 3", f: "64 × 16 × 16", c: 64, l: 16, p: 18560, d: "convolución 3 × 3, BatchNorm, ReLU y pooling" },
    { n: "Bloque 4", f: "128 × 8 × 8", c: 128, l: 8, p: 73984, d: "convolución 3 × 3, BatchNorm, ReLU y pooling" },
    { n: "Media global", f: "128", c: 128, l: 1, p: 0, d: "la media de cada uno de los 128 canales" },
    { n: "Densa", f: "64", c: 64, l: 1, p: 8256, d: "ReLU y dropout del 50 %" },
    { n: "Salida", f: "1", c: 1, l: 1, p: 65, d: "sigmoide: la probabilidad" },
  ];
  const paso = 1100 / capas.length;
  capas.forEach((q, i) => {
    const cx = paso * i + paso / 2, alto = 24 + 110 * Math.sqrt(q.l / 256), ancho = 10 + 46 * Math.sqrt(q.c / 128);
    const y0 = 95 - alto / 2;
    s("rect", { x: cx - ancho / 2, y: y0, width: ancho, height: alto, rx: 2,
                style: i === 1 ? "fill:none;stroke:#ffd23f;stroke-dasharray:4 3;stroke-width:1.5"
                     : q.p ? "fill:var(--acento);fill-opacity:.12;stroke:var(--acento);stroke-opacity:.5"
                     : "fill:var(--texto-3);fill-opacity:.06;stroke:var(--texto-3);stroke-opacity:.6;stroke-dasharray:3 3",
                "data-tip": `<b>${q.n}</b><br>${q.f}<br>${q.d}${q.p ? `<br>${q.p.toLocaleString("es-ES")} parámetros` : ""}` }, svg);
    if (i < capas.length - 1) s("path", { d: `M${cx + ancho / 2 + 10} 95 l${paso - ancho - 24} 0`, style: "stroke:var(--borde-fuerte);stroke-width:1.5;marker-end:url(#flecha)" }, svg);
    s("text", { class: "marca-texto", x: cx, y: 178, "text-anchor": "middle", texto: q.n }, svg);
    s("text", { x: cx, y: 194, "text-anchor": "middle", texto: q.f }, svg);
    s("text", { x: cx, y: 210, "text-anchor": "middle", texto: q.p ? `${q.p.toLocaleString("es-ES")} parámetros` : "sin parámetros" }, svg);
  });
  const defs = s("defs", {}, svg), mk = s("marker", { id: "flecha", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
  s("path", { d: "M0 0 L10 5 L0 10 z", style: "fill:var(--borde-fuerte)" }, mk);
  $("arq-nota").textContent = `${D.modelo.entrenamiento.parametros.toLocaleString("es-ES")} parámetros por red. Pasa el ratón por cada capa para ver qué hace.`;
}

// ---- validacion y test de los cuatro modelos ----
function dibujarRocComparada(conj) {
  const svg = $("g-roc-comp"); vaciar(svg);
  const c = D.comparacion[conj];
  const marco = [40, 306, 10, 270], x = escala(0, 1, marco[0], marco[1]), y = escala(0, 1, marco[3], marco[2]);
  ejes(svg, { x, y, xt: [0, 0.5, 1], yt: [0, 0.5, 1], fx: (v) => coma(v, 1), fy: (v) => coma(v, 1), xl: "1 − especificidad", yl: "sensibilidad", marco });
  s("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), style: "stroke:var(--texto-3);stroke-dasharray:2 3;opacity:.7" }, svg);
  [...MODELOS].reverse().forEach((n) => {
    const j = MODELOS.indexOf(n), puntos = curvaRocDe(c.filas.map((f) => [f[j], f[4]]));
    const discontinua = n === "solo_cohorte";
    const linea = s("polyline", { points: puntos.map(([a, b]) => `${x(a)},${y(b)}`).join(" "), class: discontinua ? "" : "trazo",
      style: `fill:none;stroke:${COLOR_MODELO[n]};stroke-width:${n === "recorte" ? 2.8 : 1.8}${discontinua ? ";stroke-dasharray:5 4" : ""}` }, svg);
    if (!discontinua) linea.setAttribute("pathLength", "1");   // para animar el trazado al cambiar de conjunto
  });
  const u = c.con_el_umbral;
  s("circle", { cx: x(1 - u.especificidad), cy: y(u.sensibilidad), r: 6, style: "fill:var(--anotacion-texto);stroke:var(--consola);stroke-width:2.5",
                "data-tip": `<b>CNN con el umbral ${coma(c.umbral, 3)}</b><br>sensibilidad ${pct(u.sensibilidad)}, especificidad ${pct(u.especificidad)}` }, svg);
  $("l-roc-comp").innerHTML = MODELOS.map((n) => {
    const m = c.modelos[n];
    return `<span><i class="${n === "solo_cohorte" ? "discontinua" : "linea"}" style="--c:${COLOR_MODELO[n]}"></i>${NOMBRE_MODELO[n]} <b>${coma(m.auc, 3)}</b></span>`;
  }).join("");
}

function dibujarBosque(conj) {    // AUC con su IC95 en validacion y en test, por modelo
  const svg = $("g-bosque"); vaciar(svg);
  const V = D.comparacion.validacion, T = D.comparacion.test;
  const marco = [136, 404, 22, 250], x = escala(0.4, 0.85, marco[0], marco[1]);
  const ge = s("g", { class: "eje" }, svg), gr = s("g", { class: "rejilla" }, svg);
  for (const v of [0.4, 0.5, 0.6, 0.7, 0.8]) { s("line", { x1: x(v), x2: x(v), y1: marco[2], y2: marco[3] }, gr); s("text", { x: x(v), y: marco[3] + 16, "text-anchor": "middle", texto: coma(v, 1) }, ge); }
  s("line", { x1: x(0.5), x2: x(0.5), y1: marco[2], y2: marco[3], style: "stroke:var(--texto-3);stroke-dasharray:4 3" }, svg);
  s("text", { class: "titulo-eje", x: (marco[0] + marco[1]) / 2, y: marco[3] + 32, "text-anchor": "middle", texto: "AUC por paciente, con su intervalo de confianza del 95 %" }, svg);
  s("text", { x: marco[0], y: 12, texto: "círculo hueco: validación (1.097)    círculo lleno: test (176)" }, svg);
  const alto = (marco[3] - marco[2]) / MODELOS.length;
  MODELOS.forEach((n, i) => {
    const yc = marco[2] + alto * (i + 0.5);
    if (i) s("line", { x1: 8, x2: marco[1], y1: marco[2] + alto * i, y2: marco[2] + alto * i, style: "stroke:var(--borde)" }, svg);
    s("text", { class: n === "recorte" ? "marca-texto" : "", x: 8, y: yc + 4, texto: NOMBRE_MODELO[n] }, svg);
    [[V, -9, "validacion"], [T, 9, "test"]].forEach(([c, dy, nombre]) => {
      const m = c.modelos[n], yy = yc + dy, activo = nombre === conj;
      s("line", { x1: x(limitar(m.auc_ic95[0], 0.4, 0.85)), x2: x(limitar(m.auc_ic95[1], 0.4, 0.85)), y1: yy, y2: yy,
                  style: `stroke:${COLOR_MODELO[n]};stroke-width:${activo ? 2.4 : 1.6};opacity:${activo ? 1 : 0.55}` }, svg);
      s("circle", { cx: x(m.auc), cy: yy, r: activo ? 5.5 : 4.5,
                    style: nombre === "test" ? `fill:${COLOR_MODELO[n]};stroke:var(--consola);stroke-width:1.5;opacity:${activo ? 1 : 0.6}`
                                             : `fill:var(--consola);stroke:${COLOR_MODELO[n]};stroke-width:2;opacity:${activo ? 1 : 0.6}`,
                    "data-tip": `<b>${NOMBRE_MODELO[n]}</b>, ${nombre === "test" ? "test" : "validación"}<br>AUC ${coma(m.auc, 3)} (IC95 ${coma(m.auc_ic95[0], 2)} a ${coma(m.auc_ic95[1], 2)})<br>AP ${coma(m.ap, 3)}, Brier ${coma(m.brier, 4)}` }, svg);
    });
  });
}

function dibujarCohortesComparadas(conj) {
  const svg = $("g-cohortes-comp"); vaciar(svg);
  const c = D.comparacion[conj], cohortes = ["duke", "spy1", "spy2"], n3 = ["recorte", "clinico", "combinado"];
  const marco = [40, 1290, 14, 140], y = escala(0.2, 1, marco[3], marco[2]);
  const ge = s("g", { class: "eje" }, svg), gr = s("g", { class: "rejilla" }, svg);
  for (const v of [0.2, 0.5, 0.8, 1]) { s("line", { x1: marco[0], x2: marco[1], y1: y(v), y2: y(v) }, gr); s("text", { x: marco[0] - 6, y: y(v) + 4, "text-anchor": "end", texto: coma(v, 1) }, ge); }
  s("line", { x1: marco[0], x2: marco[1], y1: y(0.5), y2: y(0.5), style: "stroke:var(--texto-3);stroke-dasharray:4 3" }, svg);
  const grupo = (marco[1] - marco[0]) / cohortes.length, ancho = 64;
  cohortes.forEach((k, i) => {
    const d = c.cohortes[k], centro = marco[0] + grupo * (i + 0.5);
    s("text", { class: "marca-texto", x: centro, y: marco[3] + 18, "text-anchor": "middle", texto: NOMBRE_COHORTE[k] }, svg);
    s("text", { x: centro, y: marco[3] + 34, "text-anchor": "middle", texto: `${d.pacientes} pacientes, ${d.pcr} con pCR` }, svg);
    n3.forEach((n, j) => {
      const m = d[n], xb = centro + (j - 1) * (ancho + 10) - ancho / 2, auc = limitar(m.auc, 0.2, 1);
      s("rect", { x: xb, y: y(auc), width: ancho, height: Math.max(1, y(0.2) - y(auc)), rx: 1, style: `fill:${COLOR_MODELO[n]};opacity:.85`,
                  "data-tip": `<b>${NOMBRE_COHORTE[k]}</b>, ${NOMBRE_MODELO[n]}<br>AUC ${coma(m.auc, 3)} (IC95 ${coma(m.auc_ic95[0], 2)} a ${coma(m.auc_ic95[1], 2)})` }, svg);
      s("text", { x: xb + ancho / 2, y: y(auc) - 4, "text-anchor": "middle", texto: coma(m.auc, 2) }, svg);
    });
  });
}

function pintarComparacion(conj) {
  estado.conjunto = conj;
  document.querySelectorAll("#selector-conjunto button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.conjunto === conj)));
  const c = D.comparacion[conj];
  dibujarRocComparada(conj);
  dibujarBosque(conj);
  dibujarCohortesComparadas(conj);
  const u = c.con_el_umbral;
  $("t-matriz").innerHTML = matrizHtml(u.VN, u.FP, u.FN, u.VP);
  const tarjeta = (k, v) => `<div><span>${k}</span><b>${v}</b></div>`;
  $("t-metricas").innerHTML = tarjeta("sensibilidad", pct(u.sensibilidad)) + tarjeta("especificidad", pct(u.especificidad)) +
    tarjeta("precisión", pct(u.precision)) + tarjeta("precisión media", coma(u.precision_media, 2)) +
    tarjeta("recall medio", coma(u.recall_medio, 2)) + tarjeta("F1 media", coma(u.f1_equilibrada, 2));
  $("t-titulo").textContent = `La CNN con el umbral ${coma(c.umbral, 3)}, en ${conj === "test" ? "el test" : "validación"} (${miles(c.pacientes_n)} pacientes)`;
  const frase = (k) => {
    const d = c.diferencias[k], excluye = d.ic95[0] > 0 || d.ic95[1] < 0;
    return `<b>${conSigno(d.diferencia)}</b> (IC95 de ${conSigno(d.ic95[0])} a ${conSigno(d.ic95[1])}, ${excluye ? "significativo" : "no significativo"})`;
  };
  $("t-notas").innerHTML = `<li>Lo que la imagen añade a los datos clínicos: ${frase("imagen + clinico - clinico")}.</li>
    <li>La CNN frente a solo saber el hospital: ${frase("imagen - solo cohorte")}.</li>
    <li>La CNN frente a los datos clínicos: ${frase("imagen - clinico")}.</li>`;
}

function montarModelo() {
  const m = D.modelo, v = m.metricas_validacion, e = m.entrenamiento, T = D.comparacion && D.comparacion.test;
  $("kpis").innerHTML = T
    ? `En las <b>${T.pacientes_n} pacientes del test</b>, que el modelo nunca vio, la CNN alcanza un AUC de <b>${coma(T.modelos.recorte.auc)}</b>
       (IC95 de ${coma(T.modelos.recorte.auc_ic95[0], 2)} a ${coma(T.modelos.recorte.auc_ic95[1], 2)}), en línea con validación:
       <b>${coma(v.auc)}</b> con 1.097 pacientes. Con el umbral ${coma(m.umbral)}, su precisión media y su recall medio son
       <b>${coma(T.con_el_umbral.precision_media, 2)}</b> y <b>${coma(T.con_el_umbral.recall_medio, 2)}</b>. Son ${m.redes.length} redes de
       ${miles(e.parametros)} parámetros, entrenadas desde cero.`
    : `En validación, con 1.097 pacientes, la CNN alcanza un AUC de <b>${coma(v.auc)}</b>.`;
  dibujarArquitectura();
  if (D.comparacion && D.comparacion.test && D.comparacion.validacion) {
    document.querySelectorAll("#selector-conjunto button").forEach((b) => b.addEventListener("click", () => pintarComparacion(b.dataset.conjunto)));
    pintarComparacion("test");
  } else $("comparacion").hidden = true;
  ROC = curvaRocDe(D.validacion.pacientes.map(([p, y]) => [p, y]));
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
  $("ficha-entrenamiento").innerHTML = fila("Optimizador", `${e.optimizador}, tasa ${tasaLegible(e.lr)}`) + fila("Lote", e.lote) +
    fila("Pérdida", "entropía cruzada binaria, sin ponderar") +
    fila("Parada", `AUC de validación en media móvil de 3 épocas, paciencia 10, máximo ${e.epocas_max}`) +
    fila("Épocas elegidas", `de ${Math.min(...ep)} a ${Math.max(...ep)}, media ${coma(mediaDe(ep), 1)}`) + fila("Semillas", e.semillas.join(", ")) +
    fila("Calibración", `Platt, a = ${coma(m.calibracion.a, 3)} y b = ${coma(m.calibracion.b, 3)}`) +
    fila("Entrenado con", `${e.dispositivo_entrenamiento}, ${coma(e.segundos_por_epoca, 1)} s por época`) + fila("PyTorch", e.version_torch);
}

// ------------------------------------------------------------------------------------------------ vista Verificacion
function montarVerificacion() {
  const fila = (k, val) => `<dt>${k}</dt><dd class="mono">${val}</dd>`;
  $("ficha-checksum").innerHTML = fila("Checksum PyTorch", D.modelo.checksum_pytorch) +
    fila("Checksum web esperado", W ? W.checksum_web : "—") + fila("Checksum web calculado", '<span id="cs-calc">—</span>') +
    fila("ONNX frente a PyTorch", W ? `${W.onnx_frente_a_pytorch.toExponential(1)} en el logit` : "—") +
    "";
  if (DEMO) {
    $("sello").querySelector("h3").textContent = "No disponible en la maqueta";
    $("sello").querySelector(".texto-suave").textContent = "La verificación necesita los modelos. Funciona en la página publicada.";
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
  sello.querySelector("h3").textContent = ok ? "La página reproduce el cálculo de Python" : "Hay diferencias con Python";
  sello.querySelector(".texto-suave").textContent = `${W.ejemplos.length} cortes. Diferencia máxima de probabilidad: ${peor.toExponential(1)} (tolerancia ${TOLERANCIA.toExponential(0)}).`;
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
  if (!D) { document.body.insertAdjacentHTML("afterbegin", '<div class="franja-demo">Falta datos/datos.js: ejecuta python datos_web.py</div>'); return; }
  $("franja-demo").hidden = !DEMO;
  document.querySelectorAll(".menu button").forEach((b) => b.addEventListener("click", () => cambiarVista(b.dataset.vista)));
  const inicial = location.hash.slice(1); if (["predecir", "modelo", "verificacion", "acerca"].includes(inicial)) cambiarVista(inicial);
  $("tema").addEventListener("click", () => {
    const claro = document.documentElement.dataset.theme !== "light";
    if (claro) document.documentElement.dataset.theme = "light"; else delete document.documentElement.dataset.theme;
    try { localStorage.setItem("tema", claro ? "light" : ""); } catch (e) {}
  });
  // ranuras: un fichero va a su fase; varios a la vez se reparten por el nombre (o van al lote)
  document.querySelectorAll('.ranura input[type="file"]').forEach((input) => {
    const fase = input.dataset.fase, ranura = input.closest(".ranura");
    input.addEventListener("change", async () => {
      const lista = input.files;
      if (!lista.length) { ponerFichero(fase, null, null); return; }
      if (lista.length > 1) { await repartirFicheros(lista); input.value = ""; return; }
      ponerFichero(fase, await lista[0].arrayBuffer(), lista[0].name);
    });
    ["dragenter", "dragover"].forEach((t) => input.addEventListener(t, () => ranura.classList.add("encima")));
    ["dragleave", "drop"].forEach((t) => input.addEventListener(t, () => ranura.classList.remove("encima")));
  });
  const zona = $("lote-zona"), loteInput = $("lote-input");
  loteInput.addEventListener("change", async () => { if (loteInput.files.length) await repartirFicheros(loteInput.files); loteInput.value = ""; });
  ["dragenter", "dragover"].forEach((t) => loteInput.addEventListener(t, () => zona.classList.add("encima")));
  ["dragleave", "drop"].forEach((t) => loteInput.addEventListener(t, () => zona.classList.remove("encima")));
  $("predecir").addEventListener("click", predecir);
  $("limpiar").addEventListener("click", limpiar);
  // visor
  document.querySelectorAll("[data-fase-visor]").forEach((b) => b.addEventListener("click", () => { if (estado.animando) pararSecuencia(); mostrarFase(b.dataset.faseVisor); }));
  $("par-comparar").addEventListener("change", () => { estado.par = $("par-comparar").value.split("|"); if (estado.fases) dibujarVisor(null, "COMPARAR"); });
  $("play").addEventListener("click", animarSecuencia);
  $("ver-recorte").addEventListener("change", () => { if (estado.fases) mostrarFase(estado.faseVisor); });
  $("ver-lupa").addEventListener("change", () => { if (!$("ver-lupa").checked) $("lupa").hidden = true; });
  const marcoVisor = $("marco-visor");
  marcoVisor.addEventListener("pointerdown", (ev) => {
    if (estado.faseVisor !== "COMPARAR" || !estado.fases) return;
    estado.arrastrando = true; marcoVisor.setPointerCapture(ev.pointerId); moverCortina(ev);
  });
  marcoVisor.addEventListener("pointermove", (ev) => {
    estado.puntero = { clientX: ev.clientX, clientY: ev.clientY };
    if (estado.arrastrando) moverCortina(ev); else dibujarLupa(estado.puntero);
  });
  ["pointerup", "pointercancel"].forEach((t) => marcoVisor.addEventListener(t, () => { estado.arrastrando = false; }));
  marcoVisor.addEventListener("pointerleave", () => { estado.puntero = null; $("lupa").hidden = true; });
  window.addEventListener("resize", () => {
    if (estado.fases && ["COMPARAR", "TRES"].includes(estado.faseVisor) && !estado.animando) dibujarVisor(null, estado.faseVisor);
  });
  // historial y verificacion
  $("descargar").addEventListener("click", descargarHistorial);
  $("vaciar").addEventListener("click", () => { estado.historial = []; $("historial").innerHTML = '<tr><td colspan="9" class="vacio">Aún no hay predicciones en esta sesión.</td></tr>'; $("descargar").disabled = true; $("vaciar").disabled = true; });
  $("verificar").addEventListener("click", verificar);
  // portada
  $("portada-empezar").addEventListener("click", () => cerrarPortada("predecir"));
  $("portada-modelo").addEventListener("click", () => cerrarPortada("modelo"));
  $("abrir-portada").addEventListener("click", abrirPortada);
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !$("portada").hidden) cerrarPortada(null); });
  $("r-umbral").textContent = `${coma(D.modelo.umbral, 3)}, equilibrado`;
  $("r-dispositivo").textContent = DEMO ? "maqueta, sin modelos" : "CPU de tu navegador (WebAssembly, ONNX Runtime)";
  const en = D.modelo.entrenamiento, mv = D.modelo.metricas_validacion, mt = D.comparacion && D.comparacion.test;
  $("r-entrenamiento").textContent = `${D.modelo.redes.length} redes (3 semillas × 5 folds) de ${miles(en.parametros)} parámetros; ` +
    `${en.optimizador}, tasa ${tasaLegible(en.lr)}, lote ${en.lote}, recorte central ${en.recorte} × ${en.recorte}, ` +
    `parada temprana con paciencia 10`;
  const u = mv.con_el_umbral;
  $("r-metricas").textContent = `AUC ${coma(mv.auc)} en validación (1.097 pacientes)` +
    (mt ? ` y ${coma(mt.modelos.recorte.auc)} en el test (${mt.pacientes_n}); ` : "; ") +
    `con el umbral, en validación: sensibilidad ${pct(u.sensibilidad)} y especificidad ${pct(u.especificidad)}`;
  $("r-checksum").textContent = DEMO ? "— (maqueta)" : "calculando…";
  $("r-checksum-pt").textContent = D.modelo.checksum_pytorch;
  cineticaVacia();
  resultadoInicial();
  montarGaleria();
  montarModelo();
  montarVerificacion();
  cargarModelo();
  let vista = null;
  try { vista = sessionStorage.getItem("portada"); } catch (e) {}
  if (!vista && !inicial && !new URLSearchParams(location.search).has("sinportada")) abrirPortada();
}

iniciar();
