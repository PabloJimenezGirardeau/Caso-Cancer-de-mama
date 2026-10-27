// Generado por exportar_web.py: no editar a mano.
window.WEB = {
 "generado": "2026-10-27T06:36:13",
 "modelos": [
  "bn_normal_c128_lr3e-05_fold0.onnx",
  "bn_normal_c128_lr3e-05_fold1.onnx",
  "bn_normal_c128_lr3e-05_fold2.onnx",
  "bn_normal_c128_lr3e-05_fold3.onnx",
  "bn_normal_c128_lr3e-05_fold4.onnx",
  "bn_normal_c128_lr3e-05_s1_fold0.onnx",
  "bn_normal_c128_lr3e-05_s1_fold1.onnx",
  "bn_normal_c128_lr3e-05_s1_fold2.onnx",
  "bn_normal_c128_lr3e-05_s1_fold3.onnx",
  "bn_normal_c128_lr3e-05_s1_fold4.onnx",
  "bn_normal_c128_lr3e-05_s2_fold0.onnx",
  "bn_normal_c128_lr3e-05_s2_fold1.onnx",
  "bn_normal_c128_lr3e-05_s2_fold2.onnx",
  "bn_normal_c128_lr3e-05_s2_fold3.onnx",
  "bn_normal_c128_lr3e-05_s2_fold4.onnx"
 ],
 "opset_onnx": 17,
 "onnx_frente_a_pytorch": 2.2649765014648438e-06,
 "mapa": "salida 'mapa' [N, 16, 16]: Grad-CAM con signo del ultimo bloque (x 256) sobre el cuadrado central; positivo = evidencia de pCR = 1",
 "mapa_frente_a_gradcam": 1.6894290481613141e-07,
 "checksum_web": "0e339bf9eb817f7e",
 "checksum_web_como": "SHA-256 de los 15 .onnx (en este orden) y de 'umbral|a|b'; primeros 16 caracteres",
 "checksum_pytorch": "19c0ffd39e140ce7",
 "nota_ejemplos": "Pacientes de ENTRENAMIENTO (nunca del test). La mayoria de los 15 modelos las vieron al entrenar, asi que sus predicciones no miden el rendimiento: sirven para comprobar que la pagina reproduce a Python.",
 "ejemplos": [
  {
   "sample_id": "ISPY2-798604_z013",
   "patient_id": "ISPY2-798604",
   "fold": 3,
   "pCR": 1,
   "rutas": {
    "PRE": "ejemplos/ISPY2-798604_z013_PRE.png",
    "EARLY": "ejemplos/ISPY2-798604_z013_EARLY.png",
    "LATE": "ejemplos/ISPY2-798604_z013_LATE.png"
   },
   "probabilidad_python": 0.37164437770843506,
   "clase_python": "pCR",
   "avisos_python": []
  },
  {
   "sample_id": "Breast_MRI_614_z010",
   "patient_id": "Breast_MRI_614",
   "fold": 1,
   "pCR": 1,
   "rutas": {
    "PRE": "ejemplos/Breast_MRI_614_z010_PRE.png",
    "EARLY": "ejemplos/Breast_MRI_614_z010_EARLY.png",
    "LATE": "ejemplos/Breast_MRI_614_z010_LATE.png"
   },
   "probabilidad_python": 0.29969218373298645,
   "clase_python": "no pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-287300_z024",
   "patient_id": "ISPY2-287300",
   "fold": 1,
   "pCR": 1,
   "rutas": {
    "PRE": "ejemplos/ISPY2-287300_z024_PRE.png",
    "EARLY": "ejemplos/ISPY2-287300_z024_EARLY.png",
    "LATE": "ejemplos/ISPY2-287300_z024_LATE.png"
   },
   "probabilidad_python": 0.31943413615226746,
   "clase_python": "pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-472799_z024",
   "patient_id": "ISPY2-472799",
   "fold": 3,
   "pCR": 1,
   "rutas": {
    "PRE": "ejemplos/ISPY2-472799_z024_PRE.png",
    "EARLY": "ejemplos/ISPY2-472799_z024_EARLY.png",
    "LATE": "ejemplos/ISPY2-472799_z024_LATE.png"
   },
   "probabilidad_python": 0.3684746325016022,
   "clase_python": "pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-810965_z020",
   "patient_id": "ISPY2-810965",
   "fold": 0,
   "pCR": 1,
   "rutas": {
    "PRE": "ejemplos/ISPY2-810965_z020_PRE.png",
    "EARLY": "ejemplos/ISPY2-810965_z020_EARLY.png",
    "LATE": "ejemplos/ISPY2-810965_z020_LATE.png"
   },
   "probabilidad_python": 0.2654161751270294,
   "clase_python": "no pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-797499_z009",
   "patient_id": "ISPY2-797499",
   "fold": 0,
   "pCR": 1,
   "rutas": {
    "PRE": "ejemplos/ISPY2-797499_z009_PRE.png",
    "EARLY": "ejemplos/ISPY2-797499_z009_EARLY.png",
    "LATE": "ejemplos/ISPY2-797499_z009_LATE.png"
   },
   "probabilidad_python": 0.33497461676597595,
   "clase_python": "pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-327133_z058",
   "patient_id": "ISPY2-327133",
   "fold": 3,
   "pCR": 0,
   "rutas": {
    "PRE": "ejemplos/ISPY2-327133_z058_PRE.png",
    "EARLY": "ejemplos/ISPY2-327133_z058_EARLY.png",
    "LATE": "ejemplos/ISPY2-327133_z058_LATE.png"
   },
   "probabilidad_python": 0.2278079092502594,
   "clase_python": "no pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-342996_z025",
   "patient_id": "ISPY2-342996",
   "fold": 4,
   "pCR": 0,
   "rutas": {
    "PRE": "ejemplos/ISPY2-342996_z025_PRE.png",
    "EARLY": "ejemplos/ISPY2-342996_z025_EARLY.png",
    "LATE": "ejemplos/ISPY2-342996_z025_LATE.png"
   },
   "probabilidad_python": 0.37058669328689575,
   "clase_python": "pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-554577_z026",
   "patient_id": "ISPY2-554577",
   "fold": 0,
   "pCR": 0,
   "rutas": {
    "PRE": "ejemplos/ISPY2-554577_z026_PRE.png",
    "EARLY": "ejemplos/ISPY2-554577_z026_EARLY.png",
    "LATE": "ejemplos/ISPY2-554577_z026_LATE.png"
   },
   "probabilidad_python": 0.25949421525001526,
   "clase_python": "no pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ACRIN-6698-654064_z035",
   "patient_id": "ACRIN-6698-654064",
   "fold": 4,
   "pCR": 0,
   "rutas": {
    "PRE": "ejemplos/ACRIN-6698-654064_z035_PRE.png",
    "EARLY": "ejemplos/ACRIN-6698-654064_z035_EARLY.png",
    "LATE": "ejemplos/ACRIN-6698-654064_z035_LATE.png"
   },
   "probabilidad_python": 0.3150944411754608,
   "clase_python": "no pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-836114_z016",
   "patient_id": "ISPY2-836114",
   "fold": 1,
   "pCR": 0,
   "rutas": {
    "PRE": "ejemplos/ISPY2-836114_z016_PRE.png",
    "EARLY": "ejemplos/ISPY2-836114_z016_EARLY.png",
    "LATE": "ejemplos/ISPY2-836114_z016_LATE.png"
   },
   "probabilidad_python": 0.29354795813560486,
   "clase_python": "no pCR",
   "avisos_python": []
  },
  {
   "sample_id": "ISPY2-125418_z041",
   "patient_id": "ISPY2-125418",
   "fold": 2,
   "pCR": 0,
   "rutas": {
    "PRE": "ejemplos/ISPY2-125418_z041_PRE.png",
    "EARLY": "ejemplos/ISPY2-125418_z041_EARLY.png",
    "LATE": "ejemplos/ISPY2-125418_z041_LATE.png"
   },
   "probabilidad_python": 0.31537294387817383,
   "clase_python": "no pCR",
   "avisos_python": []
  }
 ]
};
