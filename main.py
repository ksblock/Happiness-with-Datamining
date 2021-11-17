import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def main():
    """ main function"""
    #출력 설정
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    pd.set_option('display.max_seq_items', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    #read execl data
    raw_data = pd.read_csv('dataset/2015.csv')

    #separate index and factor score
    index = raw_data.iloc[:,0:3]
    data = raw_data.drop(['Country','Region','Happiness Rank','Happiness Score', 'Standard Error'],axis = 1)

    #calculate coefficient correlation
    corr = data.corr(method='pearson')

    df = pd.DataFrame(data)
    df_scaled = StandardScaler().fit_transform(df)

    #data description
    means = df.mean()
    sd = df.std()

    print('Means')
    print(means)
    print('Standard Deviation')
    print(sd)
    #add normality test, basic visualization

    #dimension reduction
    pca = PCA(n_components = 3)
    pca.fit(df_scaled)
    df_pca = pca.transform(df_scaled)

    pca_columns = ['pca_component_1', 'pca_component_2', 'pca_component_3']
    df_pca = pd.DataFrame(df_pca, columns=pca_columns)

    '''fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(df_pca['pca_component_1'], df_pca['pca_component_2'], df_pca['pca_component_3'])
    #plt.show()'''

    #data clustering
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df_pca)

    result = df_pca.copy()
    result['cluster'] = kmeans.labels_

    #visualization
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['pca_component_1'], result['pca_component_2'], result['pca_component_3'], c = result['cluster'], cmap='rainbow')
    plt.show()

    output = pd.concat([index, result], axis=1)
    output.to_csv('dataset/2015_result.csv')

if __name__ == '__main__':
    main()