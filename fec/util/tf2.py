import tensorflow as tf


def load_frozen_graph(file_path, input_layer_names, output_layer_names):
    with tf.io.gfile.GFile(file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    func_imported = tf.compat.v1.wrap_function(_imports_graph_def, [])
    graph_imported = func_imported.graph

    return func_imported.prune(
        tf.nest.map_structure(
            graph_imported.as_graph_element,
            input_layer_names
        ),
        tf.nest.map_structure(
            graph_imported.as_graph_element,
            output_layer_names
        )
    )
