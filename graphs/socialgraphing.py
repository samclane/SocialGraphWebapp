import os
from collections import namedtuple
from ast import literal_eval

from sklearn.neural_network import MLPClassifier

os.environ.items()  # STOP REMOVING THIS IMPORT. I USE IT I SWEAR!
import networkx as nx
import pandas
from pylab import *
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, label_binarize
from sqlalchemy import create_engine


Metrics = namedtuple('Metrics', 'cross_val accuracy class_report conf_matrix')


def preprocess(df: pandas.DataFrame):
    # Evaluate strings as lists
    df['present'] = df['present'].apply(literal_eval).apply(list).apply(lambda x: [str(y) for y in x])

    # Remove members that only appear less than N times
    # It's kinda interesting to see everyone on here
    df = df[df.groupby('member').member.transform(len) > 2]  # .reset_index()

    # Remove empty records
    # Note: Doesn't seem to help
    # df = df[df["present"].map(len) > 0]

    return df


def svm_param_selection(X, y, nfolds):
    Cs = np.linspace(0.001, 10, 5)
    gammas = np.linspace(0.01, 1, 5)
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


def encode_data(df: pandas.DataFrame):
    # Encode "present" users as OneHotVectors
    mlb = MultiLabelBinarizer()
    print("Encoding data...")
    mlb.fit(df["present"] + df["member"].apply(str).apply(lambda x: [x]))

    # Encode user labels as ints
    enc = LabelEncoder()
    flat_member_list = mlb.classes_
    enc.fit(flat_member_list)
    X, y = mlb.transform(df["present"]), enc.transform(df["member"].apply(str))
    return X, y, mlb, enc, flat_member_list


def build_svc(X_train, y_train):
    print("Training svm...")
    # params = svm_param_selection(X, y, 2)
    # params = {'C': 5.0, 'gamma': 0.01}
    # print(params)
    svc = svm.SVC(C=5.0, gamma=0.01, kernel="linear", probability=True)
    svc.fit(X_train, y_train)
    return svc


def build_mlp(X_train, y_train):
    print("Training perceptron...")
    mlp = MLPClassifier(solver='lbfgs')  # Using lbfgs we lose a bit of accuracy but convergence is much, much faster
    mlp.fit(X_train, y_train)
    return mlp


def train_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.50, random_state=42, stratify=y)
    # If we use PCA here, clf gets better at stronger labels, worse at weaker ones. Also FSFTIWD gets popularity
    # destroyed
    # clf = build_svc(X_train, y_train)
    clf = build_mlp(X_train, y_train)
    return clf, (X_train, X_test, y_train, y_test)


def get_metrics(enc, mlb, clf, X, y):
    print("Generating metrics...")
    cross_val = "\n".join(
        [f"<b>Run {n+1}</b>: {val}" for n, val in enumerate(cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=3),
                                                                            n_jobs=-1))])
    y_metric = mlb.transform([[y] for y in enc.inverse_transform(y)])
    X_metric = mlb.transform([[u] for u in enc.inverse_transform(clf.predict(X))])
    accuracy = accuracy_score(y_metric, X_metric)
    # Dictionary magic time
    id_to_enc = dict(zip(enc.classes_, enc.transform(enc.classes_)))
    real_name_map = {**id_to_enc, **get_namedict_from_sql()}
    actual_dict = {}
    for x in id_to_enc.items():
        actual_dict[x[1]] = str(real_name_map[x[0]])
    class_report = classification_report(y_metric, X_metric, target_names=list(actual_dict.values()), output_dict=True)
    #conf_matrix = '\n'.join(
    #    [f"<b>{name}</b>: {line}" for name, line in zip(actual_dict.values(), confusion_matrix(y, clf.predict(X)))])
    conf_matrix = {name: line for name, line in zip([c[0] for c in class_report.items() if c[1]["support"] > 0], confusion_matrix(y, clf.predict(X)))}
    return Metrics(cross_val, accuracy, class_report, conf_matrix)


def get_namedict_from_sql():
    df = pandas.read_sql('SELECT * FROM member_names;', create_engine(os.environ.get('DATABASE_URL')),
                         index_col='member')
    namemap = {}
    for uid in df.index:
        namemap[str(uid)] = df.loc[uid]['username']
    return namemap


def graph_data(binarizer: MultiLabelBinarizer, encoder: LabelEncoder, classifier, member_list, percentile: float = 0,
               name_file=None):
    print("Building graph...")
    social_graph = nx.DiGraph()
    social_graph.add_nodes_from(encoder.classes_)
    for u in classifier.classes_:
        u = encoder.inverse_transform([u])[0]
        others = list(binarizer.classes_)
        others.remove(u)
        # Create outgoing edges
        for o in others:
            vec = binarizer.transform([[o]])
            if encoder.transform([o]) in classifier.classes_:
                prob_map = {encoder.inverse_transform([classifier.classes_[n]])[0]: classifier.predict_proba(vec)[0][n]
                            for
                            n in range(len(classifier.classes_))}
                weight = float(prob_map[u])  # * (1 + member_list.value_counts(normalize=True)[o])
            else:
                weight = 0
            social_graph.add_edge(o, u, weight=weight)
    # Prune useless nodes
    for n in list(social_graph.nodes):
        if social_graph.in_degree(weight='weight')[n] == 0 and social_graph.out_degree(weight='weight')[n] == 0:
            social_graph.remove_node(n)

    plt.subplot(121)
    if name_file:
        mapping = {k: v for (k, v) in name_file.items() if k in social_graph.nodes}
    else:
        mapping = {k: v for (k, v) in get_namedict_from_sql().items() if k in social_graph.nodes}
    nx.relabel_nodes(social_graph, mapping, copy=False)
    popularity_list = '\n'.join(f"<b>{str(x[0])}</b>: {str(x[1])}" for x in
                                 sorted(social_graph.in_degree(weight='weight'), key=lambda x: x[1], reverse=True))
    # pos = nx.circular_layout(social_graph)
    pos = nx.fruchterman_reingold_layout(social_graph)
    edges, weights = zip(*[i for i in
                           sorted(nx.get_edge_attributes(social_graph, 'weight').items(), key=lambda x: x[1])[
                           int(len(nx.get_edge_attributes(social_graph, 'weight').items()) * percentile):]])
    nx.draw(social_graph, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.get_cmap("winter"), with_labels=True,
            arrowstyle='fancy')
    plt.title('Social Graph')
    print("Done. Showing graph.")
    return social_graph, popularity_list


def compute_roc_auc(n_classes, y_test, y_score):
    # Compute ROC curve and ROC area for each class
    print("Computing AUC...")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def plot_roc_auc(fpr, tpr, roc_auc):
    plt.subplot(122)
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label="Random Guess Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")


def init_svm_graphs(filename=None, view_percentile=0, names=None, save_file=None):
    close('all')
    if filename:
        df = pandas.read_csv(filename)
    else:
        try:
            print("Reading from db...")
            df = pandas.read_sql('SELECT * FROM member_data;', create_engine(os.environ.get('DATABASE_URL')),
                                 index_col='timestamp')
            if df.empty:
                raise Exception("`member_data` returned empty frame. Something must have gone wrong with the database.")
            print("Done reading.")
        except Exception as e:
            print("Error reading from DB {}".format(e))
            try:
                df = pandas.read_csv('graphs/data.csv')
            except FileNotFoundError:
                df = pandas.read_csv('data.csv')
    df = preprocess(df)
    X, y, mlb, enc, member_list = encode_data(df)
    clf, split_data = train_data(X, y)
    X_train, X_test, y_train, y_test = split_data

    # Generate Social Graph
    graph, popularity = graph_data(mlb, enc, clf, member_list, view_percentile, name_file=names)

    try:
        y_score = clf.decision_function(X_test)
    except AttributeError:
        print("Soft classifier found. Using predict_proba instead")
        y_score = clf.predict_proba(X_test)

    y_test_onehot = label_binarize(y_test, clf.classes_)
    fpr, tpr, roc_auc = compute_roc_auc(len(clf.classes_), y_test_onehot, y_score)
    plot_roc_auc(fpr, tpr, roc_auc)
    plt.plot()
    # Save File
    if save_file:
        save_as_graphml(graph, save_file)

    return (popularity,) + get_metrics(enc, mlb, clf, X_test, y_test)


if __name__ == "__main__":
    import argparse


    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("filename",
                            help="Name of file in the current working directory that contains the dataframe "
                                 "info", type=str)
        parser.add_argument("-nf", "--noise_floor", type=float, help="Cull edges below a certain weight. Only affects "
                                                                     "plot view.")
        parser.add_argument("-n", "--names", help="Name of csv file mapping Discord IDs and Usernames")
        parser.add_argument("-s", "--save_file", type=str, help="Filename to save as .graphml")
        return parser.parse_args()


    def save_as_graphml(graph, filename):
        print("Saving graph...")
        for node in graph.nodes():
            graph.node[node]['label'] = node
        nx.write_graphml(graph, filename)
        print("Done.")


    args = get_args()
    x = init_svm_graphs(args.filename, args.noise_floor, args.names, args.save_file)
    print('\n--------------------\n'.join([str(y) for y in x]))
    plt.show()
