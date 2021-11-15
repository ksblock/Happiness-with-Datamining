# Happiness with Datamining

## 기본 흐름

1. 데이터 읽기
2. 상관관계 분석
3. dimension reduction
4. clustering - KMeans
5. visualization

### 현재 구현 내용

1. 데이터 읽기
	- dataset 폴더에 있는 엑셀 파일 읽기

2. 상관관계 분석
	- 각 factor 간 상관계수 분석

3. dimension reduction
	- noramlization
	- PCA (n_component == 3)
	- dataframe에 index 부여
	-pca_result.png

4. clustering
	- KMeans (n_cluster == 5)
	- dataframe에 cluster index 추가

5. visualization
	- 3차원 산점도(scatter)
	- cluster에 따라 색 부여
	- kmeans_result.png

### 추가 사항

1. 데이터 읽기
	- 전체 연도 파일을 읽어야 하는지 결정
	- data description 작성 (mean, SD 등)

2. 상관관계 분석
	- 시각화
	- 활용방안 모색

3. dimension reduction
	- 국가명 유지 가능한지 확인
	- 7개 factor 중 일부만 pca 가능한지 확인

4. clustering
	- 국가명 유지 가능한지 확인
	- KMeans 외 다른 방법 가능한지
	- n_cluster 개수 조절

5. visualization
	- 3차원 그래프 각도 조절 가능?
	- 범례 넣기 (plt.legend())
