import pandas as pd
import pickle
import matplotlib.pyplot as plt

DEFAULT_FILE_PATH = "all_results.pkl"

def draw_all_pairs(df):
    names = [n.replace("_result", "") for n in df.columns if n.endswith('_result')]
    for i, name_1 in enumerate(names):
        for name_2 in names[i+1:]:
            draw_from_dataframe(df, name_1, name_2)

def draw_queries_from_df(df, name_1=None, name_2=None, draw_errors=True):
    draw_from_dataframe(df, name_1, name_2, draw_errors, 'queries')


def draw_time_from_df(df, name_1=None, name_2=None, draw_errors=True):
    draw_from_dataframe(df, name_1, name_2, draw_errors, 'queries')


def draw_from_dataframe(df, name_1=None, name_2=None, draw_errors=True, draw_param='queries'):
    '''
    :param df: with columns: exp_name and foreach algorithm have: name_result, name_queries
    :param name_1: Optional, x axis, if None use an algorithm from df.columns
    :param name_2: Optional, y axis, if None use an algorithm from df.columns
    :return:
    '''
    names =  [n.replace("_result", "") for n in df.columns if n.endswith('_result')]

    if len(names) < 2:
        raise ValueError('data frame is invalid')
    if name_1 is not None and name_2 is not None:
        x_alg = name_1
        y_alg = name_2
        assert df.loc[:, x_alg + "_result"] is not None
        assert df.loc[:, y_alg + "_result"] is not None
    else:
        x_alg = names[0]
        y_alg = names[1]

    x_name = x_alg + "_" + draw_param
    y_name = y_alg + "_" + draw_param
    df[x_name] = df[x_name].astype('int32')
    df[y_name] = df[y_name].astype('int32')
    # Filter only to rows both algorithms proved

    if draw_errors:
        max_non_error = max(df_weighted.max()[[x_alg + "_queries", y_alg + "_queries"]])
        df_filter = df
        df_filter.loc[df_filter[x_alg + "_result"] == False, x_alg + "_queries"] = max_non_error + 100
        df_filter.loc[df_filter[y_alg + "_result"] == False, y_alg + "_queries"] = max_non_error + 100
    else:
        df_filter = df.loc[(df[x_alg + '_result']) & (df[y_alg + '_result'])]
    df_filter.plot.scatter(x=x_name, y=y_name)

    max_val = max(df_filter[x_name].max(), df_filter[y_name].max())
    rgb = [i/256 for i in (145, 40, 230)]
    plt.xlim(0, max_val + 50)
    plt.ylim(0, max_val + 50)
    plt.plot(plt.xlim(), plt.ylim(), '--', color= rgb + [0.6])
    if draw_errors:
        plt.hlines(max_non_error, 0, max_non_error, linestyles='dashed', color='orange')
        plt.vlines(max_non_error, 0, max_non_error, linestyles='dashed', color='orange')

    plt.title("Number of Queries")
    plt.xlabel(x_alg.replace('_big', ''))
    plt.ylabel(y_alg.replace('_big', ''))
    plt.show()

if __name__ == "__main__":
    # # draw_all_pairs(df)
    weighted_exp_summary = "results_model_20classes_rnn4_fc32_epochs40weighted_relative_weighted_big_absolute_weighted_absolute.pkl"
    df_weighted = pickle.load(open(weighted_exp_summary, "rb"))
    draw_from_dataframe(df_weighted, 'weighted_relative', 'weighted_big_absolute')

    # relative_exp_summary = "results_model_20classes_rnn4_fc32_epochs40random_relative_iterate_relative_weighted_relative.pkl"
    # df_relative = pickle.load(open(relative_exp_summary, "rb"))
    # draw_from_dataframe(df_relative, 'weighted_relative', 'random_relative')
    #
    # absolute_exp_summary = "results_model_20classes_rnn4_fc32_epochs40iterate_big_absolute_weighted_big_absolute_result.pkl"
    # df_absolute = pickle.load(open(absolute_exp_summary, "rb"))
    # draw_from_dataframe(df_absolute, 'iterate_big_absolute', 'weighted_big_absolute')
