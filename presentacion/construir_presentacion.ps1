# Construye la presentacion de la defensa sobre la plantilla de la UAX: portada, 5 diapositivas de contenido
# (las de GUIA.md, apartado C7) con su guion en las notas, y el cierre. Usa PowerPoint (COM), que escribe el fichero.
#
# Uso, desde la raiz del proyecto:
#   powershell -ExecutionPolicy Bypass -File presentacion\construir_presentacion.ps1 [carpeta para PNG de revision]
# Salida: presentacion\Presentacion_pCR.pptx y presentacion\Presentacion_pCR.pdf

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
$rec = Join-Path $PSScriptRoot "recursos"
$salida = Join-Path $PSScriptRoot "Presentacion_pCR.pptx"
$pdf = Join-Path $PSScriptRoot "Presentacion_pCR.pdf"
$render = if ($args.Count -gt 0) { $args[0] } else { $null }

# ---- estilo: el de la plantilla (blanco y azul UAX) ----
$AZUL = "0760AB"; $AZUL_TINTE = "E9F1FA"; $TINTA = "16191F"; $TEXTO = "3B4350"; $GRIS = "6B7380"
$GRIS_CLARO = "D5DBE3"; $PANEL = "F2F4F7"; $BLANCO = "FFFFFF"
$F = "Segoe UI"; $FS = "Segoe UI Semibold"

function C([string]$h) {
    if ($h -notmatch '^[0-9A-Fa-f]{6}$') { throw "Color no valido [$h], llamado desde $((Get-PSCallStack)[1].Location)" }
    [Convert]::ToInt32($h.Substring(0, 2), 16) + 256 * [Convert]::ToInt32($h.Substring(2, 2), 16) + 65536 * [Convert]::ToInt32($h.Substring(4, 2), 16)
}
function Frag([string]$cadena, [double]$tam, [string]$col = $TEXTO, [string]$letra = $F) { @{ t = $cadena; s = $tam; c = $col; f = $letra } }

function PonerTexto($sh, $runs, [int]$alin = 1, [int]$anc = 1, [double]$inter = 0, [double]$despues = 0, [double]$margen = 0) {
    $sh.TextFrame2.AutoSize = 0
    $tf = $sh.TextFrame
    $tf.WordWrap = -1; $tf.AutoSize = 0
    $tf.MarginLeft = $margen; $tf.MarginRight = $margen; $tf.MarginTop = $margen; $tf.MarginBottom = $margen
    $tf.VerticalAnchor = $anc
    $tf.TextRange.Text = ""
    foreach ($r in $runs) {
        if (-not $r.t) { continue }
        $t = $tf.TextRange.InsertAfter($r.t)
        $t.Font.Name = $r.f; $t.Font.Size = $r.s; $t.Font.Color.RGB = (C $r.c); $t.Font.Bold = 0; $t.Font.Italic = 0
    }
    $pf = $tf.TextRange.ParagraphFormat
    $pf.Alignment = $alin; $pf.SpaceBefore = 0; $pf.SpaceAfter = $despues
    $pf.Bullet.Visible = 0
    if ($inter -gt 0) { $pf.LineRuleWithin = -1; $pf.SpaceWithin = $inter }
}

function Texto($s, [double]$x, [double]$y, [double]$w, [double]$h, $runs, [int]$alin = 1, [int]$anc = 1, [double]$inter = 0, [double]$despues = 0) {
    $sh = $s.Shapes.AddTextbox(1, $x, $y, $w, $h)
    PonerTexto $sh $runs $alin $anc $inter $despues 0
    $sh.TextFrame2.AutoSize = 0
    $sh.Left = $x; $sh.Top = $y; $sh.Width = $w; $sh.Height = $h
    $sh
}

function Caja($s, [double]$x, [double]$y, [double]$w, [double]$h, [string]$relleno, [double]$radio = 0, [string]$linea = "", [double]$grosor = 0.75, [int]$tipo = 0) {
    if ($tipo -eq 0) { $tipo = $(if ($radio -gt 0) { 5 } else { 1 }) }
    $sh = $s.Shapes.AddShape($tipo, $x, $y, $w, $h)
    if ($radio -gt 0 -and $tipo -eq 5) { $sh.Adjustments.Item(1) = [Math]::Min(0.5, $radio / [Math]::Min($w, $h)) }
    if ($relleno) { $sh.Fill.Visible = -1; $sh.Fill.Solid(); $sh.Fill.ForeColor.RGB = (C $relleno); $sh.Fill.Transparency = 0 } else { $sh.Fill.Visible = 0 }
    if ($linea) { $sh.Line.Visible = -1; $sh.Line.ForeColor.RGB = (C $linea); $sh.Line.Weight = $grosor } else { $sh.Line.Visible = 0 }
    $sh.Shadow.Visible = 0
    $sh
}

function Linea($s, [double]$x1, [double]$y1, [double]$x2, [double]$y2, [string]$color, [double]$g = 1, [int]$guion = 1, [int]$flecha = 0) {
    $l = $s.Shapes.AddLine($x1, $y1, $x2, $y2)
    $l.Line.ForeColor.RGB = (C $color); $l.Line.Weight = $g; $l.Line.DashStyle = $guion
    if ($flecha) { $l.Line.EndArrowheadStyle = 2; $l.Line.EndArrowheadLength = 1; $l.Line.EndArrowheadWidth = 2 }
    $l
}

function Imagen($s, [string]$ruta, [double]$x, [double]$y, [double]$w, [double]$h, [double]$radio = 0) {
    $p = $s.Shapes.AddPicture($ruta, 0, -1, $x, $y, $w, $h)
    if ($radio -gt 0) { $p.AutoShapeType = 5; $p.Adjustments.Item(1) = [Math]::Min(0.5, $radio / [Math]::Min($w, $h)) }
    $p
}

# Cabecera comun: circulo con el numero del punto de la guia, de que trata y el titulo (la conclusion de la diapositiva)
function Cabecera($s, [int]$n, [string]$etiqueta, [string]$titulo) {
    $c = Caja $s 48 31 22 22 $AZUL 0 "" 0.75 9
    PonerTexto $c @((Frag "$n" 11 $BLANCO $FS)) 2 3 0 0 0
    $null = Texto $s 78 31 600 22 @((Frag $etiqueta 12 $GRIS)) 1 3
    $null = Texto $s 48 60 864 40 @((Frag $titulo 26 $TINTA $FS)) 1 1
}

function Chip($s, [double]$x, [double]$y, [string]$t, [bool]$activo) {
    $relleno = $(if ($activo) { $AZUL_TINTE } else { "" }); $borde = $(if ($activo) { "" } else { $GRIS_CLARO })
    $sh = Caja $s $x $y 40 20 $relleno 10 $borde 0.75
    PonerTexto $sh @((Frag $t 10 $(if ($activo) { $AZUL } else { $GRIS }) $(if ($activo) { $FS } else { $F }))) 2 3 0 0 0
    $tf = $sh.TextFrame
    $tf.MarginLeft = 8; $tf.MarginRight = 8; $tf.MarginTop = 2; $tf.MarginBottom = 2
    $tf.WordWrap = 0; $tf.AutoSize = 1
    $sh.Left = $x; $sh.Top = $y
    $sh
}

function Notas($s, [string]$t) {
    $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = (($t.Trim() -replace "`r`n", "`r") -replace "`n", "`r")
}

function Num([string]$v) { [double]::Parse($v, [Globalization.CultureInfo]::InvariantCulture) }

# ---------------------------------------------------------------------------------------------------------------
Copy-Item (Join-Path $rec "plantilla_uax.pptx") $salida -Force
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($salida, 0, 0, 0)
try {
    # estructura: portada, 5 diapositivas de contenido (la plantilla blanca con el logo abajo a la derecha), cierre
    $pres.Slides.Item(2).Delete()
    for ($i = 0; $i -lt 4; $i++) { $null = $pres.Slides.Item(2).Duplicate() }

    # ======================================= portada =======================================
    $s = $pres.Slides.Item(1)
    $tit = $s.Shapes.Item("CuadroTexto 1")
    $tit.Top = 186; $tit.Height = 170
    PonerTexto $tit @(
        (Frag "Predicción de la pCR`ren resonancia DCE de mama" 32 $BLANCO $FS),
        (Frag "`rPablo Jiménez Girardeau" 20 $BLANCO $F),
        (Frag "`rAprendizaje automático" 14 $BLANCO $F)) 3 1 0.95 8
    $fecha = $s.Shapes.Item("CuadroTexto 10")
    $fecha.Width = 300; $fecha.Left = $tit.Left + $tit.Width - 300; $fecha.Top = 338; $fecha.Height = 22
    PonerTexto $fecha @((Frag "Curso 2026-2027" 13 $BLANCO $F)) 3 1
    Notas $s @"
Buenos días. Soy Pablo Jiménez Girardeau y presento el caso BreastDCEDL: predecir la respuesta completa a la quimioterapia con una red convolucional propia y la resonancia de antes del tratamiento.
"@

    # ================================ 1. problema clinico y objetivo ================================
    $s = $pres.Slides.Item(2)
    Cabecera $s 1 "Problema clínico y objetivo" "¿Se puede anticipar la pCR antes del tratamiento?"
    $fases = @(@("PRE", "antes del contraste", 48), @("EARLY", "contraste temprano", 176), @("LATE", "contraste tardío", 304), @("REALCE", "EARLY − PRE", 444))
    foreach ($fz in $fases) {
        $null = Imagen $s (Join-Path $rec "fase_$($fz[0]).png") $fz[2] 122 118 118 7
        $nombre = $(if ($fz[0] -eq "REALCE") { "Realce" } else { $fz[0] })
        $null = Texto $s $fz[2] 248 118 36 @((Frag $nombre 12.5 $TINTA $FS), (Frag "`r$($fz[1])" 11 $GRIS)) 1 1
    }
    $null = Linea $s 48 294 422 294 $GRIS 1.25 1 1
    $null = Texto $s 48 300 374 16 @((Frag "tiempo desde la inyección del contraste" 10.5 $GRIS)) 1 1
    $null = Texto $s 48 336 514 60 @((Frag "El mismo corte de la paciente ISPY1_1155. Antes de programar miramos las fases y el realce: el tumor capta el contraste y se ilumina. Esta paciente alcanzó la pCR." 12 $TEXTO)) 1 1 0.95
    $null = Texto $s 590 116 200 54 @((Frag "pCR" 44 $AZUL $FS)) 1 1
    $null = Texto $s 590 172 322 22 @((Frag "Respuesta patológica completa" 15 $TINTA $FS)) 1 1
    $null = Texto $s 590 198 322 58 @((Frag "Tras la quimioterapia previa a la cirugía no queda tumor invasivo. Saberlo antes permitiría plantear desescalar el tratamiento." 12.5 $TEXTO)) 1 1 0.95
    $null = Texto $s 590 262 118 54 @((Frag "29 %" 40 $AZUL $FS)) 1 3
    $null = Texto $s 704 262 208 54 @((Frag "de las 1.273 pacientes`ralcanzan la pCR" 12.5 $TEXTO)) 1 3 0.95
    $null = Caja $s 590 336 322 112 $AZUL_TINTE 10
    $null = Texto $s 606 336 290 112 @((Frag "Objetivo" 13 $AZUL $FS), (Frag "`rUna CNN propia, entrenada desde cero, que estime la probabilidad de pCR solo con la imagen de antes del tratamiento." 12.5 $TINTA)) 1 3 0.95 4
    Notas $s @"
La pregunta es si podemos anticipar, antes del tratamiento, si una paciente alcanzará la respuesta patológica completa, la pCR: que tras la quimioterapia previa a la cirugía no quede tumor invasivo. Se asocia a un mejor pronóstico, y saberlo antes permitiría plantear desescalar el tratamiento.

Cada muestra es un corte de resonancia en tres momentos: PRE, antes del contraste; EARLY, cuando llega; y LATE, unos minutos después. Antes de programar miré las fases y el realce, EARLY menos PRE, que marca el tumor. Esta paciente alcanzó la pCR.

La alcanza un 29 % de las pacientes. El objetivo: una CNN propia, entrenada desde cero y solo con la imagen, que dé una probabilidad de pCR.

Si pregunta qué aporta predecir pCR = 1: con un modelo muy preciso se podría reducir tratamiento; con el nuestro no, por la precisión (diapositiva 4).
"@

    # ============================ 2. datos, separacion por paciente y clases ============================
    $s = $pres.Slides.Item(3)
    Cabecera $s 2 "Datos, preparación, separación por paciente y clases" "1.273 pacientes de tres cohortes, separadas por paciente"
    $null = Caja $s 48 186 150 110 $AZUL 8
    $null = Texto $s 48 186 150 110 @((Frag "1.273" 28 $BLANCO $FS), (Frag "`rpacientes" 12 $BLANCO), (Frag "`r12.703 cortes" 10.5 "D6E6F5")) 2 3 0 2
    $null = Linea $s 198 241 222 241 $GRIS 1.25
    $null = Linea $s 222 160 222 322 $GRIS 1.25
    $null = Linea $s 222 160 244 160 $GRIS 1.25 1 1
    $null = Linea $s 222 322 244 322 $GRIS 1.25 1 1
    $null = Caja $s 246 118 224 84 $AZUL_TINTE 8
    $null = Texto $s 260 118 204 84 @((Frag "1.097" 20 $AZUL $FS), (Frag "  entrenamiento" 14 $TINTA $FS), (Frag "`r10.945 cortes`rvalidación cruzada en 5 folds" 10.5 $TEXTO)) 1 3 0 1
    $folds = @("219", "219", "220", "220", "219")
    for ($i = 0; $i -lt 5; $i++) {
        $b = Caja $s (246 + $i * 46) 210 40 22 $BLANCO 5 $AZUL 0.75
        PonerTexto $b @((Frag $folds[$i] 10.5 $AZUL $FS)) 2 3 0 0 0
    }
    $null = Texto $s 246 236 224 14 @((Frag "pacientes por fold" 10 $GRIS)) 1 1
    $null = Caja $s 246 280 224 84 $PANEL 8
    $null = Texto $s 260 280 204 84 @((Frag "176" 20 $TINTA $FS), (Frag "  test" 14 $TINTA $FS), (Frag "`r1.758 cortes`rpartición oficial, evaluado una sola vez" 10.5 $TEXTO)) 1 3 0 1
    $tarjetas = @(
        @(48, "Preparación", "PNG en gris ÷ 255. Tres canales = tres momentos del contraste, no colores RGB."),
        @(264, "Unidad: la paciente", "Media de sus ~10 cortes; mediana, máximo y voto rinden igual. Ninguna paciente en dos grupos."))
    foreach ($tj in $tarjetas) {
        $null = Caja $s $tj[0] 382 206 76 $PANEL 8
        $null = Texto $s ($tj[0] + 12) 392 182 62 @((Frag $tj[1] 12 $TINTA $FS), (Frag "`r$($tj[2])" 10.5 $TEXTO)) 1 1 0.95 2
    }
    $null = Texto $s 520 118 392 20 @((Frag "Pacientes por cohorte y proporción de pCR" 13 $TINTA $FS)) 1 1
    $null = Caja $s 520 147 9 9 $AZUL 2
    $null = Texto $s 533 143 40 16 @((Frag "pCR" 10.5 $GRIS)) 1 3
    $null = Caja $s 574 147 9 9 $GRIS_CLARO 2
    $null = Texto $s 587 143 60 16 @((Frag "no pCR" 10.5 $GRIS)) 1 3
    $cohortes = @(@("Duke", "", 251, 53), @("I-SPY1", "", 139, 38), @("I-SPY2", " (con ACRIN 6698)", 883, 284))
    for ($k = 0; $k -lt 3; $k++) {
        $y0 = 174 + $k * 58
        $nom, $extra, $n, $n1 = $cohortes[$k]
        $null = Texto $s 520 $y0 392 18 @((Frag $nom 12 $TINTA $FS), (Frag $extra 10.5 $GRIS), (Frag "   $n pacientes" 10.5 $GRIS)) 1 3
        $ancho = 300 * $n / 883; $anchoPcr = $ancho * $n1 / $n
        $null = Caja $s 520 ($y0 + 22) $anchoPcr 18 $AZUL 3
        $null = Caja $s (520 + $anchoPcr + 2) ($y0 + 22) ($ancho - $anchoPcr - 2) 18 $GRIS_CLARO 3
        $pct = [Math]::Round(100 * $n1 / $n)
        $null = Texto $s (520 + $ancho + 8) ($y0 + 21) 90 20 @((Frag "$pct % pCR" 12 $AZUL $FS)) 1 3
    }
    $null = Caja $s 520 350 392 98 $AZUL_TINTE 10
    $null = Texto $s 536 350 100 98 @((Frag "71 %" 30 $AZUL $FS)) 1 3
    $null = Texto $s 640 350 258 98 @((Frag "acertaría un modelo que dijera siempre «no pCR». Por eso medimos con AUC y F1 equilibrada, no con accuracy." 12 $TINTA)) 1 3 0.95
    Notas $s @"
Son 1.273 pacientes de tres cohortes: Duke, I-SPY1 e I-SPY2, que incluye ACRIN 6698. Usé la partición oficial: 1.097 pacientes para entrenar, en 5 folds de validación cruzada, y 176 para el test, evaluado una sola vez.

Se separa por paciente: todos sus cortes van al mismo grupo, y el código lo comprueba: cero pacientes en común. La unidad de evaluación es la paciente: su probabilidad es la media de sus unos diez cortes; mediana, máximo y voto dieron lo mismo, y la media es la más estable.

Cada fase se lee en gris y se divide entre 255; las tres fases son tres canales de tiempo, no colores.

Las clases están desequilibradas: un 29 % de pCR, del 21 % en Duke al 32 % en I-SPY2. Decir siempre «no pCR» acierta el 71 %, así que mido con AUC y F1 equilibrada.

Si pregunta: z012 es la altura del corte, no un instante de tiempo.
"@

    # ===================================== 3. arquitectura =====================================
    $s = $pres.Slides.Item(4)
    Cabecera $s 3 "Arquitectura de la CNN" "Una CNN propia de 106.001 parámetros, entrenada desde cero"
    $yc = 203
    $null = Imagen $s (Join-Path $rec "fase_LATE.png") 64 150 90 90 5
    $null = Imagen $s (Join-Path $rec "fase_EARLY.png") 56 158 90 90 5
    $null = Imagen $s (Join-Path $rec "fase_PRE.png") 48 166 90 90 5
    $null = Imagen $s (Join-Path $rec "fase_EARLY.png") 190 158 90 90 5
    $q = Caja $s 212.5 180.5 45 45 "" 0 $AZUL 1.75
    $q.Line.DashStyle = 4
    $null = Linea $s 162 $yc 182 $yc $GRIS 1.25 1 1
    $null = Linea $s 288 $yc 306 $yc $GRIS 1.25 1 1
    $bloques = @(@(330, 96, 12, "8DB8E2", "16 × 64 × 64"), @(410, 72, 18, "4F8FCC", "32 × 32 × 32"), @(490, 50, 26, "0760AB", "64 × 16 × 16"), @(570, 34, 36, "054884", "128 × 8 × 8"))
    for ($i = 0; $i -lt 4; $i++) {
        $cx, $alto, $ancho, $color, $dims = $bloques[$i]
        $null = Caja $s ($cx - $ancho / 2) ($yc - $alto / 2) $ancho $alto $color 2
        if ($i -lt 3) {
            $sig = $bloques[$i + 1]
            $null = Linea $s ($cx + $ancho / 2 + 6) $yc ($sig[0] - $sig[2] / 2 - 6) $yc $GRIS_CLARO 1.25 1 1
        }
        $null = Texto $s ($cx - 39) 262 78 34 @((Frag "Bloque $($i + 1)" 11 $TINTA $FS), (Frag "`r$dims" 10.5 $GRIS)) 2 1 0.95
    }
    $null = Caja $s 646 ($yc - 30) 8 60 "054884" 2
    $null = Caja $s 731 ($yc - 20) 8 40 $AZUL 2
    $null = Caja $s 815 ($yc - 15) 30 30 $AZUL 0 "" 0.75 9
    $null = Linea $s 594 $yc 640 $yc $GRIS 1.25 1 1
    $null = Linea $s 660 $yc 725 $yc $GRIS_CLARO 1.25 1 1
    $null = Linea $s 745 $yc 809 $yc $GRIS_CLARO 1.25 1 1
    $null = Texto $s 33 262 136 50 @((Frag "Entrada" 11 $TINTA $FS), (Frag "`r3 × 256 × 256" 10.5 $GRIS), (Frag "`r3 momentos, no RGB" 10.5 $GRIS)) 2 1 0.95
    $null = Texto $s 175 262 120 34 @((Frag "Recorte central" 11 $TINTA $FS), (Frag "`r3 × 128 × 128" 10.5 $GRIS)) 2 1 0.95
    $null = Texto $s 610 262 80 34 @((Frag "Media global" 11 $TINTA $FS), (Frag "`r128" 10.5 $GRIS)) 2 1 0.95
    $null = Texto $s 690 262 90 34 @((Frag "Densa" 11 $TINTA $FS), (Frag "`r64, dropout 0,5" 10.5 $GRIS)) 2 1 0.95
    $null = Texto $s 780 262 100 34 @((Frag "Salida" 11 $TINTA $FS), (Frag "`r1 neurona, sigmoide" 10.5 $GRIS)) 2 1 0.95
    $null = Linea $s 292 304 608 304 $GRIS_CLARO 1
    $null = Linea $s 292 299 292 304 $GRIS_CLARO 1
    $null = Linea $s 608 299 608 304 $GRIS_CLARO 1
    $null = Texto $s 282 308 336 16 @((Frag "cada bloque: conv 3 × 3, BatchNorm, ReLU y max pooling 2 × 2" 10.5 $GRIS)) 2 1
    $null = Texto $s 48 338 230 20 @((Frag "Parámetros  " 11 $GRIS), (Frag "106.001" 14 $AZUL $FS)) 1 3
    $param = @(@(330, "464"), @(410, "4.672"), @(490, "18.560"), @(570, "73.984"), @(650, "0"), @(735, "8.256"), @(830, "65"))
    foreach ($pp in $param) { $null = Texto $s ($pp[0] - 35) 340 70 16 @((Frag $pp[1] 11 $TEXTO)) 2 3 }
    $porque = @(
        @("Punto de partida", "La arquitectura de referencia del caso, sin pesos preentrenados."),
        @("Tamaño", "Con la mitad de filtros aprendía peor; con dos convoluciones por bloque memorizaba."),
        @("BatchNorm", "Sin ella la red apenas aprendía: en dos folds se quedaba con la primera época."),
        @("Recorte 128 × 128", "El tumor está en el centro: mejoró los 5 folds y quitó los sobreentrenados."))
    for ($i = 0; $i -lt 4; $i++) {
        $x = 48 + $i * 219
        $null = Caja $s $x 384 207 76 $PANEL 10
        $null = Texto $s ($x + 12) 394 183 62 @((Frag $porque[$i][0] 12 $TINTA $FS), (Frag "`r$($porque[$i][1])" 10.5 $TEXTO)) 1 1 0.95 2
    }
    Notas $s @"
La red es propia y pequeña: 106.001 parámetros, sin pesos preentrenados. Entra el corte con sus tres fases, 3 × 256 × 256, y dentro de la red se recorta el centro, 128 × 128; así la app lo hace exactamente igual.

Cuatro bloques de convolución 3 × 3, BatchNorm, ReLU y max pooling, con 16, 32, 64 y 128 filtros: el mapa pasa de 64 a 8 píxeles de lado. Después, media global, una densa de 64 con dropout y una neurona con sigmoide.

Partí de la arquitectura de referencia y cada cambio lo medí con la validación. El tamaño: con la mitad de filtros aprendía peor y con dos convoluciones por bloque memorizaba. BatchNorm, porque sin ella apenas aprendía. Y el recorte central: el tumor está en el centro; mejoró los cinco folds y eliminó los modelos sobreentrenados.

Dato por si pregunta: tres cuartas partes de los parámetros están en el último bloque (73.984).
"@

    # ============================ 4. entrenamiento, decisiones y resultados internos ============================
    $s = $pres.Slides.Item(5)
    Cabecera $s 4 "Entrenamiento, decisiones y resultados internos" "Parada temprana y decisiones tomadas solo con la validación"
    $null = Texto $s 48 114 400 18 @((Frag "AUC por paciente en cada época (fold 3, semilla 42)" 12 $TINTA $FS)) 1 1
    $curva = Import-Csv (Join-Path $raiz "referencia\bn_normal_c128_lr3e-05_fold3_epocas.csv")
    $n = $curva.Count; $elegida = 21
    $pL = 82; $pT = 142; $pW = 358; $pH = 148; $yMin = 0.45; $yMax = 0.80
    $es = [Globalization.CultureInfo]::GetCultureInfo("es-ES")
    function GX([double]$e) { $pL + ($e - 1) / ($n - 1) * $pW }
    function GY([double]$v) { $pT + ($yMax - $v) / ($yMax - $yMin) * $pH }
    $xe = GX $elegida
    $null = Caja $s $xe $pT ($pL + $pW - $xe) $pH $AZUL_TINTE 0
    foreach ($v in @(0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)) {
        $yy = GY $v
        $null = Linea $s $pL $yy ($pL + $pW) $yy "E3E7EC" 0.75
        $null = Texto $s ($pL - 36) ($yy - 7) 30 14 @((Frag ($v.ToString("0.00", $es)) 10 $GRIS)) 3 3
    }
    $null = Linea $s $pL ($pT + $pH) ($pL + $pW) ($pT + $pH) $GRIS_CLARO 1
    foreach ($e in @(1, 11, 21, 31)) { $null = Texto $s ((GX $e) - 12) ($pT + $pH + 3) 24 14 @((Frag "$e" 10 $GRIS)) 2 1 }
    $null = Texto $s $pL ($pT + $pH + 15) $pW 14 @((Frag "época" 10 $GRIS)) 2 1
    $yAzar = GY 0.5
    $null = Linea $s $pL $yAzar ($pL + $pW) $yAzar $GRIS 0.75 3
    $null = Texto $s ($pL + 4) ($yAzar - 14) 60 12 @((Frag "azar" 9.5 $GRIS)) 1 1
    foreach ($serie in @(@("auc_train", "9AA3AF", 2), @("auc_val", $AZUL, 2.5))) {
        $pts = New-Object 'single[,]' $n, 2
        for ($i = 0; $i -lt $n; $i++) { $pts[$i, 0] = [single](GX ($i + 1)); $pts[$i, 1] = [single](GY (Num $curva[$i].($serie[0]))) }
        $trazo = $s.Shapes.AddPolyline($pts)
        $trazo.Fill.Visible = 0; $trazo.Line.ForeColor.RGB = (C $serie[1]); $trazo.Line.Weight = [single]$serie[2]
    }
    $null = Linea $s $xe $pT $xe ($pT + $pH) $AZUL 1.25 4
    $vElegida = Num $curva[$elegida - 1].auc_val; $tElegida = Num $curva[$elegida - 1].auc_train
    $null = Caja $s ($xe - 4.5) ((GY $vElegida) - 4.5) 9 9 $AZUL 0 $BLANCO 1.5 9
    $etqE = Texto $s ($xe - 146) ($pT + 3) 140 28 @((Frag "época elegida: $elegida" 10.5 $AZUL $FS), (Frag "`rluego 10 sin mejorar: para" 10 $GRIS)) 3 1 0.95
    $etqE.Fill.Visible = -1; $etqE.Fill.Solid(); $etqE.Fill.ForeColor.RGB = (C $BLANCO)
    $dif = ($tElegida - $vElegida).ToString("0.00", $es)
    $null = Texto $s ($xe + 5) ((GY (($tElegida + $vElegida) / 2)) - 7) 110 14 @((Frag "diferencia $dif" 10 $TINTA $FS)) 1 3
    $null = Texto $s ($pL + $pW - 92) ((GY 0.72) - 2) 90 14 @((Frag "entrenamiento" 10 "7A828E" $FS)) 3 1
    $null = Texto $s ($pL + $pW - 92) ((GY (Num $curva[$n - 1].auc_val)) + 8) 90 14 @((Frag "validación" 10 $AZUL $FS)) 3 1
    $null = Texto $s 48 326 404 44 @((Frag "Adam, tasa 3·10⁻⁵, lotes de 128, entropía cruzada; 5 folds × semillas 42, 1 y 2.`rParada: media móvil de 3 épocas, paciencia 10, máximo 60." 10.5 $TEXTO)) 1 1 0.95
    $null = Caja $s 48 376 404 84 $PANEL 10
    $null = Texto $s 62 376 376 84 @((Frag "Sobreajuste controlado" 12 $TINTA $FS), (Frag "`rEn la época elegida, entrenamiento − validación: 0,004 de media en los 15 modelos (solo 1 pasa del límite de 0,15). Descartadas por memorizar: tasa 3·10⁻⁴ y conv + conv (4 de 5 folds)." 10.5 $TEXTO)) 1 3 0.95 2
    # resultados internos (validacion)
    $null = Texto $s 480 112 216 18 @((Frag "AUC de validación por fold" 11.5 $GRIS)) 1 1
    $null = Texto $s 480 130 230 42 @((Frag "0,586 ± 0,020" 30 $TINTA $FS)) 1 1
    $null = Texto $s 480 174 216 16 @((Frag "de 0,558 a 0,605 en los 5 folds" 10.5 $GRIS)) 1 1
    $null = Texto $s 728 112 184 18 @((Frag "Todas las pacientes juntas" 11.5 $GRIS)) 1 1
    $null = Texto $s 728 130 184 42 @((Frag "0,573" 30 $TINTA $FS)) 1 1
    $null = Texto $s 728 174 184 16 @((Frag "IC 95 %: 0,53-0,61 (azar: 0,5)" 10.5 $GRIS)) 1 1
    # perdida normal frente a ponderada
    $null = Caja $s 480 204 432 116 $PANEL 10
    $null = Texto $s 494 212 404 18 @((Frag "Pérdida normal, no ponderada" 12 $TINTA $FS), (Frag "   5 folds, umbral 0,5" 10.5 $GRIS)) 1 1
    $null = Texto $s 600 236 70 16 @((Frag "normal" 10.5 $GRIS)) 2 1
    $null = Texto $s 672 236 80 16 @((Frag "ponderada" 10.5 $GRIS)) 2 1
    $filas = @(@("AUC", "0,567", "0,554"), @("Sensibilidad", "13 %", "43 %"), @("Especificidad", "90 %", "60 %"))
    for ($i = 0; $i -lt 3; $i++) {
        $yf = 254 + $i * 19
        $null = Texto $s 494 $yf 106 17 @((Frag $filas[$i][0] 11 $TEXTO)) 1 3
        $null = Texto $s 600 $yf 70 17 @((Frag $filas[$i][1] 11 $TINTA $FS)) 2 3
        $null = Texto $s 672 $yf 80 17 @((Frag $filas[$i][2] 11 $TINTA $FS)) 2 3
    }
    $null = Texto $s 768 236 132 74 @((Frag "Ordena igual: solo mueve el umbral, y el umbral se elige aparte." 11 $TEXTO)) 1 3 0.95
    # umbral con criterio clinico
    $null = Caja $s 480 332 432 128 $AZUL_TINTE 10
    $null = Texto $s 496 332 400 128 @((Frag "Umbral 0,316: máxima F1 equilibrada" 12 $AZUL $FS), (Frag "`rEl error grave sería un falso positivo: quitar tratamiento a quien lo necesita. Evitarlo exige mucha precisión, y ni con especificidad 95 % pasa del 32 %: no sirve para desescalar, así que se equilibran los dos errores. En validación: sensibilidad 41 %, especificidad 70 %." 11 $TINTA)) 1 3 0.95 3
    Notas $s @"
Entrené con Adam, tasa 3·10⁻⁵ y lotes de 128, en 5 folds con 3 semillas. La parada temprana vigila el AUC de validación: se queda con el máximo de la media móvil de 3 épocas y para tras 10 sin mejorar. En este fold, entrenamiento y validación suben juntos; desde la época 21 el entrenamiento sigue y la validación no, así que se queda con la 21. En los 15 modelos, la diferencia en la época elegida es de 0,004 de media.

Resultado interno: 0,586 ± 0,020 por fold y 0,573 con todas las pacientes juntas.

La pérdida ponderada no ordena mejor: solo sube la sensibilidad con el umbral 0,5, y el umbral lo elijo aparte. El error grave sería un falso positivo, pero el modelo no tiene precisión para desescalar, así que uso el umbral de máxima F1 equilibrada: 0,316.

Si pregunta: con el umbral de desescalar (especificidad 95 %), en el test no se detectó ninguna de las 53 pCR.
"@

    # ========================== 5. test, aplicacion web, limitaciones y conclusiones ==========================
    $s = $pres.Slides.Item(6)
    Cabecera $s 5 "Test, aplicación web, limitaciones y conclusiones" "La imagen supera al azar, pero los datos clínicos predicen más"
    $null = Texto $s 48 112 440 18 @((Frag "Test: 176 pacientes, evaluado una sola vez" 12 $TINTA $FS)) 1 1
    # AUC del test con su intervalo (punto e intervalo), frente a los datos clinicos y la bibliografia
    function AX([double]$v) { 214 + ($v - 0.40) / 0.40 * 256 }
    foreach ($v in @(0.4, 0.5, 0.6, 0.7, 0.8)) {
        $xv = AX $v
        $null = Linea $s $xv 136 $xv 224 $(if ($v -eq 0.5) { $GRIS } else { "E3E7EC" }) 0.75 $(if ($v -eq 0.5) { 3 } else { 1 })
        $etq = $(if ($v -eq 0.5) { "0,5 (azar)" } else { $v.ToString("0.0", $es) })
        $null = Texto $s ($xv - 30) 226 60 14 @((Frag $etq 10 $GRIS)) 2 1
    }
    $aucs = @(
        @("Fridman et al. 2025", 0.72, 0, 0, "4B5563", 0),
        @("Imagen + clínicos", 0.705, 0.618, 0.785, "8A93A0", 0),
        @("Solo clínicos", 0.695, 0.603, 0.778, "8A93A0", 0),
        @("CNN, solo imagen", 0.591, 0.497, 0.681, $AZUL, 1))
    for ($i = 0; $i -lt 4; $i++) {
        $nomA, $va, $lo, $hi, $col, $nuestro = $aucs[$i]
        $ya = 146 + $i * 21
        $null = Texto $s 48 ($ya - 8) 118 16 @((Frag $nomA 10.5 $(if ($nuestro) { $AZUL } else { $TEXTO }) $(if ($nuestro) { $FS } else { $F }))) 1 3
        $null = Texto $s 166 ($ya - 8) 40 16 @((Frag ($va.ToString("0.000", $es).TrimEnd("0")) 11 $(if ($nuestro) { $AZUL } else { $TINTA }) $FS)) 3 3
        if ($hi -gt 0) { $null = Linea $s (AX $lo) $ya (AX $hi) $ya $col 1.5 }
        if ($i -eq 0) { $null = Caja $s ((AX $va) - 4.5) ($ya - 4.5) 9 9 $BLANCO 0 $col 1.75 9 }
        else { $null = Caja $s ((AX $va) - 4.5) ($ya - 4.5) 9 9 $col 0 $BLANCO 1.5 9 }
    }
    $null = Texto $s 48 244 440 16 @((Frag "AUC con su IC 95 %. En validación la CNN dio 0,573: el test no bajó." 10.5 $GRIS)) 1 1
    # matriz y metricas por clase
    $null = Texto $s 96 266 58 14 @((Frag "pred. no" 10 $GRIS)) 2 1
    $null = Texto $s 156 266 58 14 @((Frag "pred. pCR" 10 $GRIS)) 2 1
    $null = Texto $s 40 282 52 28 @((Frag "real no" 10 $GRIS)) 3 3
    $null = Texto $s 40 314 52 28 @((Frag "real pCR" 10 $GRIS)) 3 3
    $celdas = @(@(98, 282, "93", $true), @(158, 282, "30", $false), @(98, 314, "37", $false), @(158, 314, "16", $true))
    foreach ($cd in $celdas) {
        $b = Caja $s $cd[0] $cd[1] 54 29 $(if ($cd[3]) { $AZUL } else { $PANEL }) 4
        PonerTexto $b @((Frag $cd[2] 15 $(if ($cd[3]) { $BLANCO } else { $TINTA }) $FS)) 2 3 0 0 0
    }
    $cols = @(@(338, "precisión"), @(392, "recall"), @(440, "F1"))
    foreach ($co in $cols) { $null = Texto $s $co[0] 266 52 14 @((Frag $co[1] 10 $GRIS)) 2 1 }
    $tabla = @(@("pCR", "0,35", "0,30", "0,32", 0), @("no pCR", "0,72", "0,76", "0,74", 0), @("media", "0,53", "0,53", "0,53", 1), @("siempre «no pCR»", "0,35", "0,50", "0,41", 2))
    for ($i = 0; $i -lt 4; $i++) {
        $ft = $tabla[$i]; $yt = 282 + $i * 16
        $colT = $(if ($ft[4] -eq 2) { $GRIS } else { $TEXTO }); $colV = $(if ($ft[4] -eq 2) { $GRIS } else { $TINTA }); $letV = $(if ($ft[4] -eq 1) { $FS } else { $F })
        $null = Texto $s 230 $yt 108 16 @((Frag $ft[0] 11 $colT)) 1 3
        for ($j = 0; $j -lt 3; $j++) { $null = Texto $s $cols[$j][0] $yt 52 16 @((Frag $ft[$j + 1] 11 $colV $letV)) 2 3 }
    }
    $null = Texto $s 48 354 440 32 @((Frag "Por cohorte: Duke 0,73, I-SPY2 0,63, I-SPY1 0,29. Calibrada: Brier 0,208 (referencia 0,211)." 10.5 $TEXTO)) 1 1 0.95
    $null = Texto $s 48 394 440 18 @((Frag "Limitaciones" 12 $TINTA $FS)) 1 1
    $lim = Texto $s 48 414 440 70 @((Frag "Señal débil: con 176 pacientes, el intervalo roza el 0,5.`rUn solo corte 2D: se pierde la forma del tumor.`rSensible al brillo y al escáner: I-SPY1 ordena al revés.`rDetecta 3 de cada 10 pCR: no sirve para decidir un tratamiento." 11 $TEXTO)) 1 1 0.95 2
    $pf = $lim.TextFrame.TextRange.ParagraphFormat
    $pf.Bullet.Visible = -1; $pf.Bullet.Character = 8226; $pf.Bullet.Font.Color.RGB = (C $AZUL)
    $lim.TextFrame.Ruler.Levels.Item(1).FirstMargin = 0; $lim.TextFrame.Ruler.Levels.Item(1).LeftMargin = 12
    # la app, con un caso fallado
    $cap = Imagen $s (Join-Path $rec "app_fallo.png") 512 112 400 250 8
    $cap.Line.Visible = -1; $cap.Line.ForeColor.RGB = (C $GRIS_CLARO); $cap.Line.Weight = 0.75
    $null = Texto $s 512 366 400 30 @((Frag "Caso fallado: ISPY1_1155 alcanzó la pCR y el modelo le da 0,313, a 0,003 del umbral. La app avisa de que la decisión es ajustada." 10.5 $GRIS)) 1 1 0.95
    $url = "https://pablojimenezgirardeau.github.io/Caso-Cancer-de-mama/"
    $enlace = Texto $s 512 400 400 62 @((Frag "pablojimenezgirardeau.github.io/Caso-Cancer-de-mama" 11.5 $AZUL $FS), (Frag "`rMisma lectura y normalización que Python (PNG → gris → ÷ 255, recorte dentro de la red): diferencia máxima 5·10⁻⁸. Uso educativo, sin validez clínica." 10.5 $TEXTO)) 1 1 0.95 2
    $tramo = $enlace.TextFrame.TextRange.Paragraphs(1)
    $tramo.ActionSettings.Item(1).Hyperlink.Address = $url
    $tramo.Font.Color.RGB = (C $AZUL)
    Notas $s @"
El test, 176 pacientes evaluadas una sola vez: AUC 0,591, algo por encima de la validación, así que elegir con la validación no infló la cifra. Con el umbral, la F1 media es 0,53, frente a 0,41 de decir siempre «no pCR». Los datos clínicos predicen más, 0,695, y añadir la imagen apenas suma; Fridman y colaboradores, los autores del conjunto, obtienen 0,72 en este mismo test.

A la derecha, un fallo real: esta paciente tuvo pCR y el modelo se queda a 0,003 del umbral; la app avisa de que la decisión es ajustada. La app lee y normaliza igual que Python: diferencia máxima 5·10⁻⁸.

Limitaciones: señal débil, un solo corte 2D y sensibilidad al brillo y al escáner, sobre todo en I-SPY1. Ahora, si quiere, cargamos las cinco muestras.

Si pregunta por el caso fallado: las 3 redes que no la vieron al entrenar (fold 3) le dan 0,352 y aciertan. El enlace de la diapositiva lleva a la app.
"@

    $pres.Save()
    $pres.SaveCopyAs($pdf, 32)
    if ($render) {
        New-Item -ItemType Directory -Force $render | Out-Null
        for ($i = 1; $i -le $pres.Slides.Count; $i++) { $pres.Slides.Item($i).Export((Join-Path $render "diapositiva$i.png"), "PNG", 1600, 900) }
    }
    "ok: $($pres.Slides.Count) diapositivas"
}
finally {
    $pres.Close()
    $ppt.Quit()
}
