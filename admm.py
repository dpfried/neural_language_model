import theano
import theano.tensor as T

class ADMMModel(object):
    def __init__(self, syntax_model, semantic_model, rho, other_params, y_init=1.0, semantic_gd_rate=0.1, syntactic_gd_rate=0.1):
        self.syntax_model = syntax_model
        self.semantic_model = semantic_model
        self.rho = rho
        self.other_params = other_params
        self.y_init = y_init

        # the lagrangian
        self.y = theano.shared(value=y_init, dtype=theano.config.floatX, name='y')
        self.semantic_gd_rate = theano.shared(value=semantic_gd_rate, dtype=theano.config.floatX, name='semantic_gd_rate')
        self.syntactic_gd_rate = theano.shared(value=syntactic_gd_rate, dtype=theano.config.floatX, name='syntactic_gd_rate')

    def admm_penalty(w, v):
        return self.y * (w - v) + self.rho / 2.0 * T.sqrt(T.dot((w - v).T, w - v))

    def _build_functions(self):
        self.syntax_update_function = self.make_theano_syntax_update()
        self.semantic_update_function = self.make_theano_semantic_update()
        self.y_update_function = self.make_theano_y_update()

    def make_theano_syntax_update(self):
        # build the update functions for w, the embeddings of the syntactic
        # model
        # these represent the embeddings from the semantic model for the good
        # and bad ngrams
        w_correct_embeddings = [T.vector(name='w_correct_embedding%i' % i) for i in range(self.syntax_model.sequence_length)]
        w_error_embeddings = [T.vector(name='w_error_embedding%i' % i) for i in range(self.syntax_model.sequence_length)]
        w_embeddings = w_correct_embeddings + w_error_embeddings

        # these represent the corresponding embeddings from the semantic model
        v_correct_embeddings = [T.vector(name='v_correct_embedding%i' % i) for i in range(self.syntax_model.sequence_length)]
        v_error_embeddings = [T.vector(name='v_error_embedding%i' % i) for i in range(self.syntax_model.sequence_length)]
        v_embeddings = v_correct_embeddings + v_error_embeddings

        w = T.concatenate(w_embeddings)
        v = T.concatenate(v_embeddings)

        cost = self.syntax_model.loss(w_correct_embeddings, w_error_embeddings) + admm_penalty(T.concatenate(w_embeddings),
                                                                                               T.concatenate(v_embeddings))

        updates = [(param, param - self.syntactic_gd_rate * T.grad(cost, param))
                   for param in self.syntax_model.params]

        dcorrect_embeddings = T.grad(cost, w_correct_embeddings)
        derror_embeddings = T.grad(cost, w_error_embeddings)

        return theano.function(inputs=w_embeddings + v_embeddings,
                               outputs=dcorrect_embeddings + derror_embeddings + [cost],
                               updates=updates)

    def make_theano_semantic_update(self):
        w1, w2, v1, v2 = [T.vector(name='%s_embedding' % name) for name in ['w1', 'w2', 'v1', 'v2']]

        w = T.concatenate([w1, w2])
        v = T.concatenate([v1, v2])

        actual_sim = T.scalar(name='semantic_similarity')

        cost = self.semantic_model.loss(v, actual_sim) + admm_penalty(w, v)

        updates = [(param, param - self.semantic_gd_rate * T.grad(cost, param))
                   for param in self.semantic_model.params]

        dv1 = T.grad(cost, v1)
        dv2 = T.grad(cost, v2)

        return theano.function(inputs=[w1, w2, v1, v2, actual_sim],
                               outputs=[dv1, dv2, cost],
                               updates=updates)

    def make_theano_y_update(self):
        w = T.vector('w')
        v = T.vector('v')
        residual = w - v
        delta_y = self.rho * residual
        inputs = [w, v]
        outputs = [residual, delta_y]
        updates = [(self.y, self.y + residual)]
        return theano.function(inputs=inputs,
                               outputs=outputs,
                               updates=updates)
