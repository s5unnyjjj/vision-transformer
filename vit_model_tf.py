

import tensorflow as tf
from einops import repeat


class VisionTransformer(tf.keras.layers.Layer):
    def __init__(self,
                 image_size=(224, 224),
                 patch_size=(16, 16),
                 hidden_size=1024,
                 mlp_size=4096,
                 heads=16,
                 layers=24,
                 classes=1000,
                 dropout_rate=0.0,
                 attn_dropout_rate=0.1):

        super(VisionTransformer, self).__init__()

        self.h, self.w = image_size[0], image_size[1]
        self.hidden_size = hidden_size
        self.heads = heads
        self.ph, self.pw = patch_size
        self.patch_num = (self.h // self.ph) * (self.w // self.pw)
        self.mlp_size = mlp_size
        self.attn_dropout_rate = attn_dropout_rate
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, hidden_size]))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_rate = dropout_rate
        self.encoder_layers = layers
        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, self.patch_num + 1, self.hidden_size]))
        self.classes = classes

    def mlp_block(self, input):
        out_dim = input.shape[-1]

        out = tf.keras.layers.Dense(self.mlp_size)(input)
        out = tf.keras.activations.gelu(out, approximate=True)
        out = tf.keras.layers.Dropout(rate=self.dropout_rate)(out)

        out = tf.keras.layers.Dense(out_dim)(out)
        out = tf.keras.layers.Dropout(rate=self.dropout_rate)(out)

        return out

    def msa(self, input):
        batch_size = tf.shape(input)[0]
        projection_dim = self.hidden_size // self.heads

        query = tf.keras.layers.Dense(self.hidden_size)(input)
        query = tf.reshape(query, (batch_size, -1, self.heads, projection_dim))

        key = tf.keras.layers.Dense(self.hidden_size)(input)
        key = tf.reshape(key, (batch_size, -1, self.heads, projection_dim))

        value = tf.keras.layers.Dense(self.hidden_size)(input)
        value = tf.reshape(value, (batch_size, -1, self.heads, projection_dim))

        score1 = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        score2 = score1/tf.math.sqrt(dim_key)

        weights = tf.keras.activations.softmax(score2, axis=-1)
        out = tf.matmul(weights, value)

        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, -1, self.hidden_size))
        out = tf.keras.layers.Dense(self.hidden_size)(out)

        return out

    def encoder_block(self, input):
        residual1 = input
        out = tf.keras.layers.LayerNormalization()(input)
        out = self.msa(out)
        out = tf.keras.layers.Dropout(rate=self.dropout_rate)(out)
        out += residual1

        residual2 = out
        out = tf.keras.layers.LayerNormalization()(out)
        out = self.mlp_block(out)
        out += residual2

        return out

    def transformer_encoder(self, input):
        out = input
        for _ in range(self.encoder_layers):
            out = self.encoder_block(out)

        out = tf.keras.layers.LayerNormalization(name='encoder_layer_norm')(out)

        return out

    def architecture(self, x):
        emb = tf.keras.layers.Conv2D(filters=self.hidden_size, kernel_size=(self.ph, self.pw),
                                     strides=(self.ph, self.pw), name="embedding")(x)
        emb = tf.reshape(emb, [emb.shape[1], emb.shape[2] * emb.shape[3], emb.shape[4]])

        cls_token = repeat(self.cls_token, b=emb.shape[0], pattern='() n d -> b n d')
        x = tf.keras.layers.Concatenate(axis=1)([cls_token, emb])

        x += self.pos_embedding
        x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x, training=True)

        x = self.transformer_encoder(x)

        final_out = tf.keras.layers.Dense(self.classes)(x)

        return final_out


if __name__ == '__main__':
    model = VisionTransformer(image_size=(224, 224),
                              patch_size=(16, 16),
                              hidden_size=1024,
                              mlp_size=4096,
                              heads=16,
                              layers=24,
                              classes=1000,
                              dropout_rate=0.1)

    x = tf.keras.Input(shape=[512, 224, 224, 3])
    out = model.architecture(x)

    vit_model = tf.keras.Model(inputs=x, outputs=out)
    vit_model.summary()
