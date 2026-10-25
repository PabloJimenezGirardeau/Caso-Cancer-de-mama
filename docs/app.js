"use strict";
/*
 * App web del caso BreastDCEDL: el modelo final (15 redes en ONNX) se ejecuta en el navegador.
 *
 * Reproduce exactamente el pipeline de Python (predecir.py):
 *   1. cada fase es un PNG de 256x256; se lee como escala de grises de 8 bits igual que Pillow convert("L")
 *      (L = (R*19595 + G*38470 + B*7471 + 0x8000) >> 16) y se divide entre 255;
 *   2. las 15 redes, en modo evaluacion (exportadas asi), dan cada una sigmoide(logit); el recorte central de 128x128
 *      va dentro de cada red;
 *   3. media de las 15 probabilidades, recorte a [1e-6, 1 - 1e-6] y calibracion de Platt: p = sigmoide(a*logit(media) + b);
 *   4. pCR = 1 si p >= umbral.
 * Las imagenes se leen como datos (nunca se ejecuta nada de lo subido) y no salen del navegador.
 */

const LADO = 256;
const RECORTE = 128;
const TAM_MAX = 5 * 1024 * 1024;           // 5 MB por PNG, como predecir.py
const FASES = ["PRE", "EARLY", "LATE"];
const TOLERANCIA_VERIFICACION = 1e-4;

const estado = {
  modelo: null,                            // datos/modelo.json
  sesiones: [],                            // 15 sesiones de onnxruntime
  checksumWeb: null,
  ejemplos: [],                            // datos/verificacion.json
  buffers: { PRE: null, EARLY: null, LATE: null },
  nombres: { PRE: null, EARLY: null, LATE: null },
  ultimo: null,                            // ultimo corte leido (Float32Array por fase)
};

class EntradaNoValida extends Error {}

const $ = (id) => document.getElementById(id);
const coma = (x, d = 3) => Number(x).toFixed(d).replace(".", ",");
const pct = (x, d = 0) => `${(100 * x).toFixed(d).replace(".", ",")} %`;
const sigmoide = (z) => 1 / (1 + Math.exp(-z));

// --------------------------------------------------------------------------------------------------------------
// Lectura y validacion de las imagenes (igual que utils_caso.cargar_fase / predecir.cargar_fase)
// --------------------------------------------------------------------------------------------------------------

function formatoDe(b) {
  if (b.length >= 8 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47 &&
      b[4] === 0x0d && b[5] === 0x0a && b[6] === 0x1a && b[7] === 0x0a) return "PNG";
  if (b[0] === 0xff && b[1] === 0xd8) return "JPEG";
  if (b[0] === 0x47 && b[1] === 0x49 && b[2] === 0x46) return "GIF";
  if (b[0] === 0x42 && b[1] === 0x4d) return "BMP";
  if (b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46) return "WEBP";
  return null;
}

function leerFase(buffer, nombre) {
  const b = new Uint8Array(buffer);
  if (b.length === 0) throw new EntradaNoValida(`${nombre}: el fichero está vacío.`);
  if (b.length > TAM_MAX) {
    throw new EntradaNoValida(`${nombre}: el fichero ocupa demasiado (${coma(b.length / 2 ** 20, 1)} MB; máximo 5 MB).`);
  }
  const formato = formatoDe(b);
  if (formato !== "PNG") {
    throw new EntradaNoValida(`${nombre}: debe ser un PNG (se ha recibido ${formato || "un formato desconocido"}).`);
  }
  let img, rgba;
  try {
    img = UPNG.decode(buffer);
  } catch (e) {
    throw new EntradaNoValida(`${nombre}: no se puede leer como imagen (el fichero está dañado).`);
  }
  if (img.width !== LADO || img.height !== LADO) {
    throw new EntradaNoValida(`${nombre}: debe medir ${LADO}x${LADO} píxeles (mide ${img.width}x${img.height}).`);
  }
  if (img.depth > 8) {
    throw new EntradaNoValida(`${nombre}: el PNG tiene ${img.depth} bits por canal; el modelo se entrenó con PNG de 8 bits.`);
  }
  try {
    rgba = new Uint8Array(UPNG.toRGBA8(img)[0]);
  } catch (e) {
    throw new EntradaNoValida(`${nombre}: no se puede leer como imagen (el fichero está dañado).`);
  }
  if (rgba.length !== LADO * LADO * 4) {
    throw new EntradaNoValida(`${nombre}: no se puede leer como imagen (datos incompletos).`);
  }
  const x = new Float32Array(LADO * LADO);
  for (let i = 0, j = 0; i < x.length; i++, j += 4) {
    const gris = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >> 16;   // Pillow convert("L")
    x[i] = gris / 255;
  }
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i])) throw new EntradaNoValida(`${nombre}: contiene valores no finitos.`);
  }
  return x;
}

function media(arr, mascara) {
  let s = 0, n = 0;
  for (let i = 0; i < arr.length; i++) if (!mascara || mascara[i]) { s += arr[i]; n++; }
  return n ? s / n : NaN;
}

function desviacion(arr) {
  const m = media(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += (arr[i] - m) ** 2;
  return Math.sqrt(s / arr.length);
}

// Los mismos avisos que predecir.avisos: no impiden predecir, pero conviene verlos.
function avisos(fases) {
  const [pre, early] = [fases.PRE, fases.EARLY];
  const lista = [];
  const tejido = new Uint8Array(pre.length);
  let n = 0;
  for (let i = 0; i < pre.length; i++) if (pre[i] > 0.1) { tejido[i] = 1; n++; }
  if (n < 100) {
    lista.push("La imagen PRE es casi toda negra: ¿es un corte de resonancia de mama?");
  } else if (media(early, tejido) <= media(pre, tejido)) {
    lista.push("En el tejido, EARLY no es más brillante que PRE (en el dataset lo es en el 100 % de las pacientes): " +
               "¿están las fases en el orden correcto?");
  }
  if (FASES.some((f) => desviacion(fases[f]) < 1e-3)) lista.push("Alguna fase es una imagen prácticamente uniforme.");
  for (const f of FASES) {
    const m = media(fases[f]);
    if (m > 0.8) {
      lista.push(`La fase ${f} es casi toda blanca (brillo medio ${coma(m, 2)}): puede ser una imagen defectuosa y la ` +
                 "predicción no es fiable.");
    }
  }
  return lista;
}

// --------------------------------------------------------------------------------------------------------------
// Modelo
// --------------------------------------------------------------------------------------------------------------

async function sha256Hex(partes) {
  const total = partes.reduce((s, p) => s + p.byteLength, 0);
  const junto = new Uint8Array(total);
  let pos = 0;
  for (const p of partes) { junto.set(new Uint8Array(p), pos); pos += p.byteLength; }
  const huella = await crypto.subtle.digest("SHA-256", junto);
  return Array.from(new Uint8Array(huella)).map((x) => x.toString(16).padStart(2, "0")).join("");
}

function ponerEstado(texto, clase) {
  $("estado-texto").textContent = texto;
  $("estado").className = `estado ${clase || ""}`;
}

async function cargarModelo() {
  try {
    const [modelo, verificacion] = await Promise.all([
      fetch("datos/modelo.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
      fetch("datos/verificacion.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    ]);
    estado.modelo = modelo;
    estado.ejemplos = verificacion.ejemplos;
    mostrarModelo();
    rellenarEjemplos();
    ort.env.wasm.numThreads = 1;           // GitHub Pages no permite hilos de WebAssembly; resultado reproducible
    const buffers = [];
    for (let i = 0; i < modelo.modelos.length; i++) {
      ponerEstado(`Cargando el modelo ${i + 1} de ${modelo.modelos.length}…`);
      const r = await fetch(`modelos/${modelo.modelos[i]}`);
      if (!r.ok) throw new Error(`modelos/${modelo.modelos[i]}: ${r.status}`);
      const buf = await r.arrayBuffer();
      buffers.push(buf);
      estado.sesiones.push(await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] }));
    }
    const texto = new TextEncoder().encode(`${modelo.umbral}|${modelo.calibracion.a}|${modelo.calibracion.b}`);
    estado.checksumWeb = (await sha256Hex([...buffers, texto.buffer])).slice(0, 16);
    const coincide = estado.checksumWeb === modelo.checksum_web;
    ponerEstado(coincide ? `Modelo listo · ${modelo.modelos.length} redes · checksum ${estado.checksumWeb}`
                         : `Modelo cargado, pero el checksum (${estado.checksumWeb}) no coincide con el esperado`,
                coincide ? "listo" : "fallo");
    $("verificar").disabled = false;
    actualizarBoton();
  } catch (e) {
    console.error(e);
    const local = location.protocol === "file:";
    ponerEstado(local ? "Abre la página desde un servidor web (GitHub Pages o «python -m http.server»), no como fichero."
                      : `No se ha podido cargar el modelo (${e.message}).`, "fallo");
  }
}

async function predecirCorte(fases) {
  const { umbral, calibracion } = estado.modelo;
  const x = new Float32Array(3 * LADO * LADO);
  FASES.forEach((f, k) => x.set(fases[f], k * LADO * LADO));        // orden de canales: PRE, EARLY, LATE
  const tensor = new ort.Tensor("float32", x, [1, 3, LADO, LADO]);
  const t0 = performance.now();
  const probs = [];
  for (const sesion of estado.sesiones) {
    const salida = await sesion.run({ imagen: tensor });
    probs.push(sigmoide(salida.logit.data[0]));
  }
  let m = probs.reduce((s, p) => s + p, 0) / probs.length;
  m = Math.min(Math.max(m, 1e-6), 1 - 1e-6);
  const p = sigmoide(calibracion.a * Math.log(m / (1 - m)) + calibracion.b);
  return { probabilidad: p, clase: p >= umbral ? "pCR" : "no pCR", media: m, probs,
           segundos: (performance.now() - t0) / 1000 };
}

// --------------------------------------------------------------------------------------------------------------
// Interfaz
// --------------------------------------------------------------------------------------------------------------

function dibujarGris(canvas, datos) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(LADO, LADO);
  for (let i = 0, j = 0; i < datos.length; i++, j += 4) {
    const v = Math.round(datos[i] * 255);
    img.data[j] = img.data[j + 1] = img.data[j + 2] = v;
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function color(t) {
  const x = Math.min(Math.max(t, 0), 1) * (INFERNO.length - 1);
  const i = Math.min(Math.floor(x), INFERNO.length - 2), f = x - i;
  return INFERNO[i].map((c, k) => Math.round(c + f * (INFERNO[i + 1][k] - c)));
}

function dibujarRealce(canvas, pre, early) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(LADO, LADO);
  let max = 1e-3;
  const r = new Float32Array(pre.length);
  for (let i = 0; i < pre.length; i++) { r[i] = Math.max(early[i] - pre[i], 0); if (r[i] > max) max = r[i]; }
  for (let i = 0, j = 0; i < r.length; i++, j += 4) {
    const [cr, cg, cb] = color(r[i] / max);
    img.data[j] = cr; img.data[j + 1] = cg; img.data[j + 2] = cb; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function dibujarRecorte(canvas) {
  if (!$("ver-recorte").checked) return;
  const ctx = canvas.getContext("2d");
  const ini = (LADO - RECORTE) / 2;
  ctx.save();
  ctx.strokeStyle = "#ffd60a";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 5]);
  ctx.strokeRect(ini + 1, ini + 1, RECORTE - 2, RECORTE - 2);
  ctx.restore();
}

function mostrarImagenes(fases) {
  $("panel-imagenes").hidden = false;
  for (const f of FASES) { dibujarGris($(`c-${f}`), fases[f]); dibujarRecorte($(`c-${f}`)); }
  dibujarRealce($("c-REALCE"), fases.PRE, fases.EARLY);
  dibujarRecorte($("c-REALCE"));
}

function mostrarResultado(r, listaAvisos) {
  const { umbral } = estado.modelo;
  $("panel-resultado").hidden = false;
  $("r-prob").textContent = coma(r.probabilidad, 3);
  const esPcr = r.clase === "pCR";
  $("r-clase").textContent = esPcr ? "pCR = 1 · respuesta completa probable" : "pCR = 0 · respuesta completa poco probable";
  $("r-clase").className = `clase ${esPcr ? "pcr" : "nopcr"}`;
  $("r-aguja").style.left = `${100 * r.probabilidad}%`;
  $("r-umbral-marca").style.left = `${100 * umbral}%`;
  $("r-umbral").textContent = `${coma(umbral, 3)} (${estado.modelo.criterio_umbral.split(":")[0]})`;
  $("r-media").textContent = coma(r.media, 4);
  $("r-dispositivo").textContent = `Navegador · onnxruntime-web ${ort.env.versions?.web || "1.19.2"} · WebAssembly (CPU)`;
  $("r-tiempo").textContent = `${coma(r.segundos * 1000, 0)} ms (15 redes)`;
  $("r-checksum").textContent = `${estado.checksumWeb}${estado.checksumWeb === estado.modelo.checksum_web ? " ✓" : " ✗"}`;
  $("r-checksum-pt").textContent = estado.modelo.checksum_pytorch;
  const maximo = Math.max(0.5, ...r.probs);
  $("r-modelos").innerHTML = r.probs.map((p, i) =>
    `<div class="barra" style="height:${(100 * p / maximo).toFixed(1)}%" title="${estado.modelo.modelos[i].replace(".onnx", "")}: ${coma(p, 3)}"></div>`
  ).join("");
  $("r-avisos").innerHTML = listaAvisos.map((a) => `<li>${a}</li>`).join("");
}

function mostrarErrores(lista) {
  const caja = $("errores");
  if (!lista.length) { caja.hidden = true; caja.innerHTML = ""; return; }
  caja.hidden = false;
  caja.innerHTML = `<ul>${lista.map((e) => `<li>${e}</li>`).join("")}</ul>`;
}

function actualizarBoton() {
  $("predecir").disabled = !(estado.sesiones.length === estado.modelo?.modelos.length &&
                             FASES.every((f) => estado.buffers[f]));
}

function ponerFichero(fase, buffer, nombre) {
  estado.buffers[fase] = buffer;
  estado.nombres[fase] = nombre;
  const zona = document.querySelector(`.zona[data-fase="${fase}"]`);
  const etiqueta = document.querySelector(`[data-nombre="${fase}"]`);
  zona.classList.remove("mal", "cargada");
  if (!buffer) { etiqueta.textContent = "Arrastra o elige un PNG"; actualizarBoton(); return; }
  try {
    leerFase(buffer, fase);
    zona.classList.add("cargada");
    etiqueta.textContent = nombre;
  } catch (e) {
    zona.classList.add("mal");
    etiqueta.textContent = e instanceof EntradaNoValida ? e.message : `${fase}: error inesperado.`;
  }
  actualizarBoton();
}

async function predecir() {
  mostrarErrores([]);
  const fases = {}, errores = [];
  for (const f of FASES) {
    try { fases[f] = leerFase(estado.buffers[f], f); }
    catch (e) { errores.push(e instanceof EntradaNoValida ? e.message : `${f}: error inesperado al leer la imagen.`); }
  }
  if (errores.length) { mostrarErrores(errores); $("panel-resultado").hidden = true; return; }
  estado.ultimo = fases;
  mostrarImagenes(fases);
  $("predecir").disabled = true;
  try {
    const r = await predecirCorte(fases);
    mostrarResultado(r, avisos(fases));
    $("panel-resultado").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    console.error(e);
    mostrarErrores([`No se ha podido calcular la predicción (${e.message}).`]);
  } finally {
    actualizarBoton();
  }
}

async function bajar(ruta) {
  const r = await fetch(ruta);
  if (!r.ok) throw new Error(`${ruta}: ${r.status}`);
  return r.arrayBuffer();
}

function rellenarEjemplos() {
  const sel = $("ejemplo");
  estado.ejemplos.forEach((e, i) => {
    const op = document.createElement("option");
    op.value = String(i);
    op.textContent = `Ejemplo ${i + 1} · ${e.sample_id} · pCR real = ${e.pCR} (paciente de entrenamiento)`;
    sel.appendChild(op);
  });
}

async function elegirEjemplo() {
  const i = $("ejemplo").value;
  if (i === "") return;
  const e = estado.ejemplos[Number(i)];
  mostrarErrores([]);
  try {
    for (const f of FASES) ponerFichero(f, await bajar(e.rutas[f]), e.rutas[f].split("/").pop());
  } catch (err) {
    mostrarErrores([`No se ha podido cargar el ejemplo (${err.message}).`]);
  }
}

function limpiar() {
  for (const f of FASES) ponerFichero(f, null, null);
  document.querySelectorAll('.zona input[type="file"]').forEach((i) => { i.value = ""; });
  $("ejemplo").value = "";
  $("panel-imagenes").hidden = true;
  $("panel-resultado").hidden = true;
  mostrarErrores([]);
}

function mostrarModelo() {
  const m = estado.modelo, e = m.entrenamiento, v = m.metricas_validacion, u = v.con_el_umbral;
  $("m-nombre").textContent = m.nombre;
  const fila = (k, val) => `<dt>${k}</dt><dd>${val}</dd>`;
  const epocas = Object.values(e.epocas_elegidas);
  $("m-entrenamiento").innerHTML = [
    fila("Red", e.red),
    fila("Parámetros", e.parametros.toLocaleString("es-ES")),
    fila("Optimizador", `${e.optimizador}, tasa de aprendizaje ${String(e.lr).replace(".", ",")}, lote ${e.lote}`),
    fila("Pérdida", e.perdida),
    fila("Parada", `${e.parada}; máximo ${e.epocas_max} épocas`),
    fila("Modelo final", `${m.modelos.length} redes: semillas ${e.semillas.join(", ")} × 5 folds`),
    fila("Épocas elegidas", `${Math.min(...epocas)} a ${Math.max(...epocas)} (media ${coma(epocas.reduce((s, x) => s + x, 0) / epocas.length, 1)})`),
    fila("Calibración", `Platt: a = ${coma(m.calibracion.a, 3)}, b = ${coma(m.calibracion.b, 3)}`),
    fila("Umbral", `${coma(m.umbral, 3)} (máxima F1 equilibrada en validación)`),
    fila("Entorno", `PyTorch ${e.version_torch}; exportado a ONNX (opset ${e.opset_onnx})`),
  ].join("");
  $("m-metricas").innerHTML = [
    fila("AUC por paciente", `${coma(v.auc)} (IC95 ${coma(v.auc_ic95[0], 2)}–${coma(v.auc_ic95[1], 2)})`),
    fila("AUC por fold", (v.auc_por_fold || []).map((x) => coma(x, 3)).join(" · ")),
    fila("Precisión media (AP)", `${coma(v.ap)} (azar 0,29)`),
    fila("Brier", `${coma(v.brier, 4)} (referencia ${coma(v.brier_referencia, 4)})`),
    fila("Con el umbral", `sensibilidad ${pct(u.sensibilidad)}, especificidad ${pct(u.especificidad)}, precisión ${pct(u.precision)}`),
    fila("F1", `pCR ${coma(u.f1_pcr, 2)} · no pCR ${coma(u.f1_no_pcr, 2)} · equilibrada ${coma(u.f1_equilibrada, 2)}`),
    fila("Matriz", `VP ${u.VP} · FN ${u.FN} · VN ${u.VN} · FP ${u.FP}`),
  ].join("");
  $("m-nota").textContent = v.nota;
}

async function verificar() {
  const caja = $("verificacion");
  $("verificar").disabled = true;
  caja.innerHTML = "<p>Verificando…</p>";
  const filas = [];
  let peor = 0, todasIguales = true;
  for (const e of estado.ejemplos) {
    try {
      const fases = {};
      for (const f of FASES) fases[f] = leerFase(await bajar(e.rutas[f]), f);
      const r = await predecirCorte(fases);
      const dif = Math.abs(r.probabilidad - e.probabilidad_python);
      peor = Math.max(peor, dif);
      const igual = dif < TOLERANCIA_VERIFICACION && r.clase === e.clase_python;
      todasIguales = todasIguales && igual;
      filas.push(`<tr><td>${e.sample_id}</td><td class="num">${e.probabilidad_python.toFixed(6)}</td>` +
                 `<td class="num">${r.probabilidad.toFixed(6)}</td><td class="num">${dif.toExponential(1)}</td>` +
                 `<td>${e.clase_python} / ${r.clase}</td><td class="${igual ? "bien" : "mal-texto"}">${igual ? "✓" : "✗"}</td></tr>`);
    } catch (err) {
      todasIguales = false;
      filas.push(`<tr><td>${e.sample_id}</td><td colspan="5" class="mal-texto">${err.message}</td></tr>`);
    }
  }
  caja.innerHTML = `<table><thead><tr><th>Corte</th><th>Python</th><th>Esta página</th><th>Diferencia</th>` +
    `<th>Clase (Python / página)</th><th></th></tr></thead><tbody>${filas.join("")}</tbody></table>` +
    `<p class="resumen-verificacion ${todasIguales ? "bien" : "mal-texto"}">` +
    (todasIguales ? `Coinciden los ${estado.ejemplos.length} ejemplos: diferencia máxima ${peor.toExponential(1)}. ` +
                    "La página reproduce el pipeline de Python."
                  : `Hay diferencias (máxima ${peor.toExponential(1)}): la página NO reproduce exactamente a Python.`) + "</p>";
  $("verificar").disabled = false;
}

// --------------------------------------------------------------------------------------------------------------
// Arranque
// --------------------------------------------------------------------------------------------------------------

document.querySelectorAll('.zona input[type="file"]').forEach((input) => {
  const fase = input.dataset.fase;
  const zona = input.closest(".zona");
  input.addEventListener("change", async () => {
    const fichero = input.files[0];
    $("ejemplo").value = "";
    if (!fichero) { ponerFichero(fase, null, null); return; }
    ponerFichero(fase, await fichero.arrayBuffer(), fichero.name);
  });
  ["dragenter", "dragover"].forEach((t) => input.addEventListener(t, () => zona.classList.add("encima")));
  ["dragleave", "drop"].forEach((t) => input.addEventListener(t, () => zona.classList.remove("encima")));
});
$("predecir").addEventListener("click", predecir);
$("limpiar").addEventListener("click", limpiar);
$("ejemplo").addEventListener("change", elegirEjemplo);
$("verificar").addEventListener("click", verificar);
$("ver-recorte").addEventListener("change", () => { if (estado.ultimo) mostrarImagenes(estado.ultimo); });

cargarModelo();
