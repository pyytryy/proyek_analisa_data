import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from babel.numbers import format_currency
sns.set(style='dark')

with st.sidebar:
    st.image ("https://raw.githubusercontent.com/pyytryy/Project_Structure_MC269D5X0767/main/Air%20polution.png")
    st.sidebar.title("Filter Data")

    selected_year = st.sidebar.slider("Select Year", 2013, 2017, (2013, 2017))
    selected_station = st.sidebar.selectbox("Select Station", ["All", "Aotizhongxin", "Changping", "Dingling", "Dongsi","Guanyuan"])


@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/dataset_df.csv")
    return df

df = load_data()


df_filtered = df[(df["year"] >= selected_year[0]) & (df["year"] <= selected_year[1])]
if selected_station != "All":
    df_filtered = df_filtered[df_filtered["station"] == selected_station]

st.title("🌍 Dashboard Polution Anylsis Beijing")
st.markdown(f"**Data Analysis for Year {selected_year} on {selected_station} Station**")

st.subheader(f"**Average Polution for Year {selected_year} on {selected_station} Station**")
col1, col2, col3,col4 = st.columns(4)
col1.metric("Average PM2.5", round(df_filtered["PM2.5"].mean(), 2))
col2.metric("Average PM10", round(df_filtered["PM10"].mean(), 2))
col3.metric("Average CO", round(df_filtered["CO"].mean(), 2))
col4.metric("Average O3", round (df_filtered ["O3"].mean(),2))

#Pertanyaan 1
#Dari 13 Stasiun di Beijing, stasiun manakah yang memiliki kadar PM2.5 tertinggi dari tahun 2015-2017
st.subheader(f"**Average PM2.5 for year {selected_year} on Station in Beijing ({selected_station} Station)**")
avg_pm25_per_station = df_filtered.groupby("station")["PM2.5"].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="station", y="PM2.5", data=avg_pm25_per_station, palette="Reds_r", ax=ax)

plt.xticks(rotation=90)
plt.xlabel("Station")
plt.ylabel("Rata-rata PM2.5 (µg/m³)")

st.pyplot(fig)

#Pertanyaan 2
#Apakah suhu berkorelasi dengan CO, dan bagaimana suhu berkorelasi dengan O3?
st.subheader(f"**Corelation Temperature with CO and O3 for year {selected_year} on {selected_station} Station**")

kolom_korelasi = ["CO", "O3", "TEMP"]
korelasi = df_filtered[kolom_korelasi].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(korelasi, annot=True, cmap="rocket", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Corelation Air Polution and Temperature ")
st.pyplot(fig)

#Pertanyaan 3
#Apakah polusi udara PM2.5, PM10 dan O3 meningkat dari tahun ke tahun?
st.subheader(f"**Air Polution from {selected_year} on {selected_station} Station**")
polusi_per_tahun = df_filtered.groupby("year")[["PM2.5","PM10","O3"]].mean()

average_df = polusi_per_tahun.reset_index()
st.write("Tren Average PM2.5, PM10, dan O3 From Years to Year")
st.table(average_df)

fig, ax=plt.subplots(figsize=(8, 5))
sns.lineplot(x=polusi_per_tahun.index, y=polusi_per_tahun["PM2.5"], marker="o", label="PM2.5")
sns.lineplot(x=polusi_per_tahun.index, y=polusi_per_tahun["PM10"], marker="^", label="PM10")
sns.lineplot(x=polusi_per_tahun.index, y=polusi_per_tahun["O3"], marker="s", label="O3")

plt.xlabel("Year")
plt.ylabel("Average Air Polution")
plt.title("Tren Average PM2.5, PM10, dan O3 From Years to Years")
plt.legend()
plt.grid(True)
st.pyplot(fig)


#Analisis Lanjutan
import gdown

data = pd.read_csv("dataset_df.csv")

data_clustering = data.copy()

# Drop kolom yang tidak dibutuhkan
data_clustering = data_clustering.drop(columns=["SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"])


fiturx = data_clustering["PM2.5"].values
fitury = data_clustering["PM10"].values

q1= np.percentile(fiturx, 25)
q2= np.percentile(fiturx, 75)

polusi_rendah = data_clustering[data_clustering["PM2.5"] < q1]
polusi_standar = data_clustering[(data_clustering["PM2.5"] >= q1) & (data_clustering["PM2.5"] < q2)]
polusi_tinggi = data_clustering[data_clustering["PM2.5"] >= q2]

centroid_rendah = [polusi_rendah["PM2.5"].mean(), polusi_rendah["PM10"].mean()]
centroid_standar = [polusi_standar["PM2.5"].mean(), polusi_standar["PM10"].mean()]
centroid_tinggi = [polusi_tinggi["PM2.5"].mean(), polusi_tinggi["PM10"].mean()]

clusters = []
for i in range(len(data_clustering)):
    point = [data_clustering["PM2.5"].iloc[i], data_clustering["PM10"].iloc[i]]

    dist_rendah = np.linalg.norm(np.array(point) - np.array(centroid_rendah))
    dist_standar = np.linalg.norm(np.array(point) - np.array(centroid_standar))
    dist_tinggi = np.linalg.norm(np.array(point) - np.array(centroid_tinggi))

    if dist_rendah < dist_standar and dist_rendah < dist_tinggi:
        cluster = 1
    elif dist_standar < dist_rendah and dist_standar < dist_tinggi:
        cluster = 2
    else:
        cluster = 3

    clusters.append(cluster)

data_with_clusters = data_clustering.copy()
data_with_clusters['cluster'] = clusters

st.subheader(f"**Clustering Stasiun Kadar Rendah (1), Kadar Standar (2), Kadar Tinggi(3)**")
st.dataframe(data_with_clusters[["PM2.5", "PM10", "cluster"]])

st.write('### Visualisasi Clustering Stasiun')
fig, ax = plt.subplots()

warna = {1: 'red', 2: 'blue', 3: 'green'}
label_cluster = {1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 3"}

for clus in [1, 2, 3]:
    cluster_data = data_with_clusters[data_with_clusters["cluster"] == clus]
    ax.scatter(cluster_data["PM2.5"], cluster_data["PM10"],
               label=label_cluster[clus], color=warna[clus], alpha=0.6)

ax.scatter([centroid_rendah[0], centroid_standar[0], centroid_tinggi[0]],
           [centroid_rendah[1], centroid_standar[1], centroid_tinggi[1]],
           marker="x", color="black", s=200, label="Centroid")

ax.set_xlabel("PM2.5")
ax.set_ylabel("PM10")
ax.set_title("Clustering Stasiun Kadar Polusi Rendah, Standar dan Tinggi")
ax.legend()
ax.grid(True)

st.pyplot(fig)
