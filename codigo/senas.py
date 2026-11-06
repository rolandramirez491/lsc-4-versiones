

from pathlib import Path

import numpy as np

from dibujo_senas import construir_muestra_completa, texto_ascii


RUTA_CODIGO = Path(__file__).resolve().parent
RUTA_PROYECTO = RUTA_CODIGO.parent

# Una sola carpeta de datos: "datos oficiales" (LSC50, LSC-54 y videos de YouTube)
RUTA_PROPIOS = RUTA_PROYECTO / "datos oficiales"

# La lista de señas va dentro de la carpeta de datos
ARCHIVO_SENAS = RUTA_PROPIOS / "mis_senas.txt"


def leer_senas():
    """Lista de (indice, nombre) en el orden de mis_senas.txt."""
    if not ARCHIVO_SENAS.exists():
        raise FileNotFoundError(f"No se encontró la lista de señas:\n{ARCHIVO_SENAS}")

    nombres = [
        linea.strip()
        for linea in ARCHIVO_SENAS.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.strip().startswith("#")
    ]

    if not nombres:
        raise ValueError(f"{ARCHIVO_SENAS.name} no tiene ninguna seña.")

    return list(enumerate(nombres))


def agregar_sena(nombre):
    """
    Agrega una seña AL FINAL de mis_senas.txt.
    Devuelve (lista de señas actualizada, mensaje de error o None).
    """
    nombre = " ".join(nombre.split())

    if not nombre:
        return leer_senas(), "ESCRIBE UN NOMBRE"

    if not nombre_carpeta(nombre).strip("_"):
        return leer_senas(), "NOMBRE NO VALIDO"

    senas = leer_senas()
    nueva = nombre_carpeta(nombre)

    for _, existente in senas:
        if nombre_carpeta(existente) == nueva:
            return senas, f"YA EXISTE: {texto_ascii(existente)}"

    contenido = ARCHIVO_SENAS.read_text(encoding="utf-8")
    if contenido and not contenido.endswith("\n"):
        contenido += "\n"

    ARCHIVO_SENAS.write_text(contenido + nombre + "\n", encoding="utf-8")

    return leer_senas(), None


def eliminar_sena(nombre):
    """Quita la seña de la lista (las muestras se mueven aparte antes)."""
    lineas = ARCHIVO_SENAS.read_text(encoding="utf-8").splitlines()
    clave = nombre_carpeta(nombre)
    quedan = [l for l in lineas if l.strip().startswith("#") or not l.strip()
              or nombre_carpeta(l.strip()) != clave]
    ARCHIVO_SENAS.write_text("\n".join(quedan) + "\n", encoding="utf-8")
    return leer_senas()


def nombre_carpeta(nombre):
    return texto_ascii(nombre).replace(" ", "_").replace("?", "").replace("!", "")


def carpeta_numerada(raiz, indice, nombre, nueva_con=4):
    """
    Carpeta NNNN_NOMBRE de una seña dentro de 'raiz'.
    Sirve con 4 dígitos (0005_ABEJA) y con los nombres viejos de 2 (05_ABEJA).
    Si el número cambió, la busca por el NOMBRE EXACTO (no confunde CASA con MI_CASA).
    Si no existe, devuelve la ruta nueva (sin crearla).
    """
    nc = nombre_carpeta(nombre)
    for digitos in (4, 2):
        exacta = raiz / f"{indice:0{digitos}d}_{nc}"
        if exacta.exists():
            return exacta
    if raiz.exists():
        for carpeta in raiz.glob(f"*_{nc}"):
            numero, _, resto = carpeta.name.partition("_")
            if carpeta.is_dir() and numero.isdigit() and resto == nc:
                return carpeta
    return raiz / f"{indice:0{nueva_con}d}_{nc}"


def carpeta_de(indice, nombre):
    """Carpeta donde se guardan las muestras de una seña: datos oficiales/NN_NOMBRE"""
    return carpeta_numerada(RUTA_PROPIOS, indice, nombre, nueva_con=2)


def contar(indice, nombre):
    carpeta = carpeta_de(indice, nombre)
    return len(list(carpeta.glob("muestra_*.npz"))) if carpeta.exists() else 0


def cargar_muestras_propias(con_brazo=False):
    """
    Lee todas tus grabaciones y arma dibujo + extras de cada una.
    La seña de cada muestra sale de su carpeta (no del archivo).

    Devuelve: trazos, validos, extras, y, senas
    Con con_brazo=True además: trazos_brazo, validos_brazo, tiene_brazo
    (tiene_brazo = la grabación guardó los codos; las viejas no los tienen).
    """
    senas = leer_senas()
    trazos, validos, extras, y = [], [], [], []
    trazos_b, validos_b, tiene_b = [], [], []

    print(f"Leyendo grabaciones de: {RUTA_PROPIOS}")

    for indice, nombre in senas:
        carpeta = carpeta_de(indice, nombre)
        archivos = sorted(carpeta.glob("muestra_*.npz")) if carpeta.exists() else []
        usadas = 0
        con_codos = 0

        for archivo in archivos:
            m = np.load(archivo, allow_pickle=True)
            hay_codos = "codo_izq" in m.files

            t, v, e, tb, vb = construir_muestra_completa(
                [m["muneca_izq"], m["muneca_der"]],
                [m["valida_izq"], m["valida_der"]],
                [m["mano_izq"], m["mano_der"]],
                [m["mano_izq_ok"], m["mano_der_ok"]],
                float(np.median(m["ancho_hombros"])),
                np.median(m["centro_hombros"], axis=0),
                [m["codo_izq"], m["codo_der"]] if hay_codos else None,
                [m["codo_izq_ok"], m["codo_der_ok"]] if hay_codos else None,
                m["cara"] if "cara" in m.files else None,
                m["cara_ok"] if "cara_ok" in m.files else None
            )

            if not v.any():
                continue

            trazos.append(t)
            validos.append(v)
            extras.append(e)
            y.append(indice)
            trazos_b.append(tb)
            validos_b.append(vb)
            tiene_b.append(hay_codos and bool(vb.any()))
            usadas += 1
            con_codos += int(tiene_b[-1])

        print(
            f"  {indice:02d} {texto_ascii(nombre):16} -> {usadas} muestras"
            f"  ({con_codos} con antebrazo)"
        )

    if not y:
        raise ValueError(
            "\nNo hay grabaciones. Graba primero con:  python grabar_mis_senas.py"
        )

    resultado = (
        np.stack(trazos).astype(np.float32),
        np.stack(validos).astype(bool),
        np.stack(extras).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        senas
    )

    if not con_brazo:
        return resultado

    return resultado + (
        np.stack(trazos_b).astype(np.float32),
        np.stack(validos_b).astype(bool),
        np.asarray(tiene_b, dtype=bool)
    )
