import mlx.core as mx
from mlx import nn
from mlx_graphs.nn import Linear

from mlx_graphs.nn import MessagePassing
from mlx_graphs.utils import scatter, degree


class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True, node_dim=0):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.node_dim = node_dim

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = Linear(input_dim * 10, output_dim)
        else:
            self.linear = Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = Linear(query_input_dim, num_relation * input_dim)
        else:
            # relation embeddings as an independent embedding matrix per each layer
            self.relation = nn.Embedding(num_relation, input_dim)

    def __call__(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)

        # input: (bs, num_nodes, dim)
        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).reshape(batch_size, self.num_relation, self.input_dim)
        else:
            # layer-specific relation features as a special embedding matrix unique to each layer
            relation = mx.repeat(self.relation.weight[None, :], batch_size, axis=0)
            #relation = self.relation.weight.expand(batch_size, -1, -1)
        if edge_weight is None:
            edge_weight = mx.ones(len(edge_type))  # TODO stream

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = self.propagate(node_features=input.transpose(1,0,2), edge_index=edge_index, 
                                message_kwargs=dict(relation=relation, boundary=boundary.transpose(1, 0, 2), edge_type=edge_type),
                                aggregate_kwargs=dict(edge_weight=edge_weight, dim_size=size),
                                update_kwargs=dict(input=input.transpose(1, 0, 2))) 
        return output


    def message(self, src_features, dst_features, relation, boundary, edge_type):
        #relation_j = relation.index_select(self.node_dim, edge_type)
        relation_j = relation[:, edge_type].transpose(1, 0, 2)

        if self.message_func == "transe":
            message = src_features + relation_j
        elif self.message_func == "distmult":
            message = src_features * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = src_features.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = mx.concatenate([message_re, message_im], axis=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = mx.concatenate([message, boundary], axis=0)  # (num_edges + num_nodes, batch_size, input_dim)

        return message

    def aggregate(self, messages, indices, edge_weight, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = mx.concatenate([indices, mx.arange(dim_size[0])]) # (num_edges + num_nodes,)
        edge_weight = mx.concatenate([edge_weight, mx.ones(dim_size[0])])
        shape = [1] * messages.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.reshape(shape)

        if self.aggregate_func == "pna":
            mean = scatter(messages * edge_weight, index, axis=self.node_dim, out_size=dim_size[0], aggr="mean")
            sq_mean = scatter(messages ** 2 * edge_weight, index, axis=self.node_dim, out_size=dim_size[0], aggr="mean")
            max = scatter(messages * edge_weight, index, axis=self.node_dim, out_size=dim_size[0], aggr="max")
            # scatter_min is not implemented in MLX-graphs
            # min = scatter(messages * edge_weight, index, axis=self.node_dim, out_size=dim_size[0], aggr="min")
            std = mx.clip(sq_mean - mean ** 2, a_min=self.eps, a_max=None).sqrt()
            features = mx.concatenate([mean[..., None], max[..., None], std[..., None]], axis=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size[0])[..., None, None]
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = mx.concatenate([mx.ones_like(scale), scale, 1 / mx.clip(scale, a_min=1e-2, a_max=None)], axis=-1)
            output = (features[..., None] * scales[:, :, None, :]).flatten(-2)
        else:
            output = scatter(messages * edge_weight, index, axis=self.node_dim, out_size=dim_size[0],
                             aggr=self.aggregate_func)

        return output


    def update_nodes(self, aggregated, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(mx.concatenate([input, aggregated], axis=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output.transpose(1, 0, 2)
