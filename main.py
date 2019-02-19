import numpy as np
from operator import itemgetter


class DecisionTreeRegressor:


    """
        A simple decision tree implementation with a near identical API to that of sci-kit learn.
        Trees are generated via the CART algorithm.
    """

    def __init__(self, max_depth, min_samples_split=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.nodes = {}
        self.node_counter = 0

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.assignments = np.array([[i, 0] for i in range(0, Y.shape[0])])

        self.n_samps, self.n_feats = X.shape
        if self.n_samps != Y.shape[0]:
            raise ValueError('Shape of target does not match featureset.')

        self.init_starting_node()

        for depth in range(1, self.max_depth + 1):
            leaves = [key for key, value in self.nodes.items() if self.nodes[key]['depth'] == depth]
            for leaf_node in leaves:
                self.split_node(leaf_node)

    def split_node(self, node_n):
        node_X = self.X[self.node_vals(node_n)]
        node_Y = self.Y[self.node_vals(node_n)]
        node_err = self.nodes[node_n]['err']
        node_depth = self.nodes[node_n]['depth']

        splits = []
        for feat_n in range(0, self.n_feats):
            for split_candidate in node_X[:, feat_n]:
                if self.validate_split(feat_n, split_candidate, node_X, node_Y):
                    valid, err, y_t, y_f, indx, alt_indx = self.eval_split(feat_n, split_candidate, node_X, node_Y)
                    splits.append((err, indx, alt_indx, y_t, y_f, feat_n, split_candidate))
        if len(splits) > 0:
            new_split = splits[min(enumerate([i[0] for i in splits]), key=itemgetter(1))[0]]
            self.create_new_node(node_n, node_depth, *new_split[1:])

    def create_new_node(self, node_n, depth, indx, alt_indx, y_t, y_f, feat_n, split_candidate):
        node_vals = self.node_vals(node_n)[0]
        self.assignments[:, 1][node_vals[indx[0]]] = self.node_counter + 1
        self.assignments[:, 1][node_vals[alt_indx[0]]] = self.node_counter + 2

        self.nodes[self.node_counter + 1] = {
            'y_hat': y_t,
            'err': self.tree_mse(self.Y[self.node_vals(self.node_counter + 1)],
                                 self.Y[self.node_vals(self.node_counter + 1)].mean()),
            'n_samples': len(indx[0]),
            'depth': depth + 1,
            'feat_n': feat_n,
            'split_val': split_candidate,
            'gte': True,
            'parent': node_n,
        }
        self.nodes[self.node_counter + 2] = {
            'y_hat': y_f,
            'err': self.tree_mse(self.Y[self.node_vals(self.node_counter + 2)],
                                 self.Y[self.node_vals(self.node_counter + 2)].mean()),
            'n_samples': len(alt_indx[0]),
            'depth': depth + 1,
            'feat_n': feat_n,
            'split_val': split_candidate,
            'gte': False,
            'parent': node_n,
        }

        self.node_counter += 2


    def validate_split(self, feat_n, split_val, node_X, node_Y):
        indx = np.where(node_X[:, feat_n] >= split_val)
        alt_indx = np.where(node_X[:, feat_n] < split_val)
        return indx[0].shape[0] >= self.min_samples_split and alt_indx[0].shape[0] >= self.min_samples_split

    def eval_split(self, feat_n, split_val, node_X, node_Y):
        indx = np.where(node_X[:, feat_n] >= split_val)
        alt_indx = np.where(node_X[:, feat_n] < split_val)
        y = node_Y[indx]
        y_alt = node_Y[alt_indx]
        y_len = len(y)
        y_alt_len = len(y_alt)
        tot = y_len + y_alt_len
        s1 = (y_len / tot) * self.tree_mse(y, y.mean())
        s2 = (y_alt_len / tot) * self.tree_mse(y_alt, y_alt.mean())
        return True, s1 + s2, y.mean(), y_alt.mean(), indx, alt_indx

    @staticmethod
    def tree_mse(y_true, y_est):
        return np.sum(np.power((y_true - y_est), 2)) / y_true.shape[0]

    def node_vals(self, node_n):
        return np.where(self.assignments[:, 1] == node_n)

    def init_starting_node(self):
        self.nodes[0] = {
            'y_hat': self.Y.mean(),
            'err': self.tree_mse(self.Y, self.Y.mean()),
            'n_samples': self.Y.shape[0],
            'depth': 1,
            'feat_n': 'NA',
            'split_val': 'NA',
            'gte': 'NA',
            'parent': 'NA',
        }

    def predict(self, X):
        children = self.find_children(0)

        while len(children) > 0:
            if self.satisfies_node(children[0], X):
                cur_node = children[0]
            else:
                cur_node = children[1]

            children = self.find_children(cur_node)

        return self.nodes[cur_node]['y_hat']

    def find_children(self, node_id):
        return [key for key, value in self.nodes.items() if self.nodes[key]['parent'] == node_id]

    def satisfies_node(self, node_n, dp):
        feat_n = self.nodes[node_n]['feat_n']
        split_val = self.nodes[node_n]['split_val']

        return dp[feat_n] >= split_val


class LinearRegressor:

    def __init__(self, eta, n_epochs=1):
        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self._validate_args()

        self._format_feature_matrix()

        # initialize weights for linear model
        self.weights = np.random.uniform(-.2, .2, (self.n_features + 1, 1))

        self._optimize()

    def update(self, X, Y)
        self.X = X
        self.Y = Y
        self._validate_args()

        self._format_feature_matrix()

        self._update(self.X, self.Y)

    def _format_feature_matrix(self):
        self.X = np.hstack((np.ones((self.n_rows, 1)), self.X))
        self.X = np.matrix(self.X)

    def _validate_args(self):
        x_shape = self.X.shape
        y_shape = self.Y.shape
        if x_shape[0] != y_shape[0]:
            raise ValueError(f'Invalid argument dimensions, {x_shape} feature array is incompatable' +
                             f' with {y_shape} target array.')
        if len(x_shape) != 2 or len(y_shape) != 2:
            raise ValueError(f'X and y Must be two-dimensional arrays. The X provided is of shape {x_shape}' +
                             f' and the Y provided is of shape {y_shape}')
        if y_shape[0] < 1 or x_shape[0] < 1:
            raise ValueError(f'Y (target) must contain at leat one value and X must contain at leat one row.')

        self._cache_args_metadata()

    def _cache_args_metadata(self):
        self.n_rows = self.X.shape[0]
        self.n_features = self.X.shape[1]

    def _cost_function(self, y_true, y_pred):
        np.sum(np.power(y_true - y_pred, 2)) / y_true.shape[0]

    def _cf_gradient(self, X, Y):
        return np.asarray(np.transpose(self.X) * (np.matrix(self.X) * self.weights - np.matrix(self.Y)))

    def _update(self, X, Y):
        grad = self._cf_gradient(X, Y)
        self.weights = self.weights - self.eta * grad

    def _optimize(self):
        for epoch in range(self.n_epochs):
            for i in range(self.n_rows):
                indx = np.random.randint(self.n_rows)
                x = self.X[indx]
                y = self.Y[indx]
                self._update(x, y)



