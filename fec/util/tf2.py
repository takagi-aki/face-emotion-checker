import tensorflow as tf


def load_graph_def(file_path, input_layer_names, output_layer_names):
    with tf.io.gfile.GFile(file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(
            import_graph.as_graph_element,
            input_layer_names
        ),
        tf.nest.map_structure(
            import_graph.as_graph_element,
            output_layer_names
        )
    )

def load_meta_graph_def(file_path, meta_path, ckpt_path, input_layer_names, output_layer_names):

    def import_multiply():
        tf.compat.v1.train.import_meta_graph(meta_path)

    wrapped_import = tf.compat.v1.wrap_function(import_multiply, [])
    import_graph = wrapped_import.graph

    tf.compat.v1.train.Saver(wrapped_import.variables).restore(
    sess=None, save_path=ckpt_path)

    return wrapped_import.prune(
        tf.nest.map_structure(
            import_graph.as_graph_element,
            input_layer_names
        ),
        tf.nest.map_structure(
            import_graph.as_graph_element,
            output_layer_names
        )
    )