import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def main():
    """ main function"""
    #출력 설정
    #2018년 아랍에메리트 정부 N/A 0으로 처리
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    pd.set_option('display.max_seq_items', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]

    for i in range (7):

        #read execl data
        raw_data = pd.read_csv('dataset/' + years[i] + '.csv')

        #separate index and factor score
        index = raw_data.iloc[:,:4]
        data = raw_data.iloc[:,4:]

        #calculate coefficient correlation
        corr = data.corr(method='pearson')
        #sns.heatmap(corr, cmap = 'Blues')
        #plt.savefig('result/' + years[i]+' corr heatmap.png')

        df = pd.DataFrame(data)
        df_scaled = StandardScaler().fit_transform(df)
        
        #data description
        means = df.mean()
        sd = df.std()
        print(df['Family'].describe())
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

        #data clustering
        kmeans_PCA = KMeans(n_clusters=5)
        kmeans_PCA.fit(df_pca)

        result_PCA = df_pca.copy()
        result_PCA['cluster'] = kmeans_PCA.labels_

        #visualization
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(result_PCA['pca_component_1'], result_PCA['pca_component_2'], result_PCA['pca_component_3'], c = result_PCA['cluster'], cmap='rainbow')
        ax.set_xlabel('pca_component_1')
        ax.set_ylabel('pca_component_2')
        ax.set_zlabel('pca_component_3')
        plt.savefig('result/' + years[i] + " PCA clustering.png")
        plt.show()

        output_PCA = pd.concat([index, result_PCA], axis=1)
        output_PCA.to_csv('dataset/' + years[i] + ' PCA result.csv')

        #sum method
        df_sum = pd.DataFrame(data)
        df_sum['Personal_sum'] = df_sum['Economy (GDP per Capita)'] + df_sum['Family'] + df_sum['Health (Life Expectancy)']
        df_sum['Country_sum'] = df_sum['Freedom'] + df_sum['Trust (Government Corruption)'] + df_sum['Generosity']
        df_sum = df_sum.drop(['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)', 'Generosity'], axis=1)
        print(df_sum.head())

        kmeans_sum = KMeans(n_clusters=5)
        kmeans_sum.fit(df_sum)

        result_sum = df_sum.copy()
        result_sum['cluster'] = kmeans_sum.labels_

        # visualization
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(result_sum['Dystopia Residual'], result_sum['Personal_sum'], result_sum['Country_sum'],
                   c=result_sum['cluster'], cmap='rainbow')
        ax.set_xlabel('Dystopia Residual')
        ax.set_ylabel('Personal_sum')
        ax.set_zlabel('Country_sum')
        plt.savefig('result/' + years[i] + " sum clustering.png")
        plt.show()

        output_PCA = pd.concat([index, result_sum], axis=1)
        output_PCA.to_csv('dataset/' + years[i] + ' sum result.csv')





if __name__ == '__main__':
    main()