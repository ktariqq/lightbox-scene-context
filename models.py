
import numpy as np
import keras
from keras import layers
from config import *

NC = len(CLASSES)


def customcnn():
    i = x = keras.Input((224, 224, 3), name="image")
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.Rescaling(1/255)(x)

    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(NC, name="logits")(x)

    return keras.Model(i, out, name="customcnn")


def transfercnn(proj=256, hid=128):
    i = keras.Input((1280,), name="center")
    x = layers.Dense(proj, activation="relu")(i)
    x = layers.Dense(hid, activation="relu")(x)
    out = layers.Dense(NC, name="logits")(x)

    return keras.Model(i, out, name="transfercnn")


def fdense(x, w, b=None, act=None):
    d = layers.Dense(np.shape(w)[1], activation=act, use_bias=b is not None, trainable=False)
    y = d(x)
    d.set_weights([np.float32(w)] + ([np.float32(b)] if b is not None else []))
    return y


def fusionhead(kind, proj=256, hid=128):
    ins = [keras.Input((1280,), name=v) for v in VIEWS]
    mk = keras.Input((3,), name="mask")
    m = layers.Reshape((3, 1))(mk)

    stk = lambda ts: layers.Concatenate(axis=1)([layers.Reshape((1, proj))(t) for t in ts])

    projected = [layers.Dense(proj, activation="relu", name=f"proj_{v}")(t) for v, t in zip(VIEWS, ins)]
    pm = layers.Multiply()([stk(projected), m])

    flat = layers.Flatten()(pm)

    if kind == "gated":
        gates = [layers.Dense(proj, activation="sigmoid", name=f"gate_{v}")(flat) for v in VIEWS]
        gm = layers.Multiply()([stk(gates), m])

        s = fdense(fdense(fdense(mk, np.ones((3, 1))), np.ones((1, 2)), [-1, -2], "relu"), [[-1.5], [1.]], [3.])

        fused = layers.Multiply()([gm, pm])
        z = layers.GlobalAveragePooling1D()(fused)
        z = layers.Multiply()([z, s])

    else:
        z = flat

    x = layers.Dense(hid, activation="relu")(z)
    out = layers.Dense(NC, name="logits")(x)

    tm = keras.Model(ins + [mk], out, name=kind)

    if kind == "concat":
        return tm, tm

    gate_mean = layers.GlobalAveragePooling1D()(layers.Permute((2, 1))(gm))
    full = keras.Model(ins + [mk], [out, gm, gate_mean], name="gated_full")

    return tm, full
