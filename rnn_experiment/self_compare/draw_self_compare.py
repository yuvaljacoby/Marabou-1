import pandas as pd
import pickle
import matplotlib.pyplot as plt

DEFAULT_FILE_PATH = "all_results.pkl"


def draw_from_dataframe(df, name_1=None, name_2=None):
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

    df[x_alg + "_queries"] = df[x_alg + "_queries"].astype('int32')
    df[y_alg + "_queries"] = df[y_alg + "_queries"].astype('int32')
    # Filter only to rows both algorithms proved
    df_filter = df.loc[(df[x_alg + '_result']) & (df[y_alg + '_result'])]
    df_filter.plot.scatter(x=x_alg+"_queries", y=y_alg+"_queries")
    plt.title("Number of Queries")
    plt.xlabel(x_alg)
    plt.ylabel(y_alg)
    plt.show()

if __name__ == "__main__":
    df = pickle.load(open(DEFAULT_FILE_PATH, "rb"))
    draw_from_dataframe(df)