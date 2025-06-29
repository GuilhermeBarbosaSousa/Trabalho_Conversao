# ============================================================
#   Análise Completa do Motor-Bomba  – 12 PNGs
# ============================================================

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import textwrap, warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Tight layout")

# ---------- 1. Cores / Estilo --------------------------------
COR_TEAL, COR_ROXO  = '#36C3B9', '#8D36C3'
COR_ROSA, COR_CINZA = '#F53D80', '#4C4C4C'
COR_CLARO, COR_GRADE= '#F5F5F5', '#d0d0d0'
plt.style.use('seaborn-v0_8-whitegrid')
cmap_grad = LinearSegmentedColormap.from_list('grad', [COR_ROXO, COR_ROSA, COR_TEAL])

# ---------- 2. Constantes do motor ---------------------------
P_SAIDA_KW, EF_NOM_MOTOR, I_NOM_A = 15.0, 0.928, 28.9
N_SINC, N_NOM, T_MAX_TN = 1800, 1770, 3.0
TORQUE_NOM = (P_SAIDA_KW*1000)/(N_NOM*2*np.pi/60)
EF_PTS  = np.array([91.7,93.1,93.5,93.0,92.4])/100
CARG_PTS= np.array([25,50,75,100,125])
ef_interp = lambda c: np.interp(c, CARG_PTS, EF_PTS)

# ---------- 3. Helpers ---------------------------------------
def safe_legend(ax, **kw):
    h,l = ax.get_legend_handles_labels()
    if h: ax.legend(h,l,**kw)

def tabela_png(df, titulo, nome, wrap=14):
    """Gera PNG de tabela, ajustando colunas e confirmando no console."""
    plt.style.use('default')
    df_disp = df.copy()
    df_disp.insert(0, df_disp.index.name or '', df_disp.index)
    df_disp.index = range(len(df_disp))
    df_disp = df_disp.map(lambda v: '\n'.join(textwrap.wrap(str(v), wrap)))

    rows, cols = df_disp.shape
    char_w = 0.12
    col_w = [max(df_disp[c].str.len().max(), len(c))*char_w for c in df_disp.columns]
    fig_w = sum(col_w)+1.4; fig_h = 0.6+0.55*rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis('off')
    ax.set_title(titulo, fontsize=18, fontweight='bold', color=COR_CINZA, pad=12)

    tbl = ax.table(cellText=df_disp.values, colLabels=df_disp.columns,
                   cellLoc='center', colWidths=col_w, bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1,1.15)
    cells = tbl.get_celld()
    for j in range(cols):  # cabeçalho
        c=cells[(0,j)]; c.set_facecolor(COR_CINZA); c.set_text_props(color='white',weight='bold'); c.set_edgecolor('white')
    for i in range(1, rows+1):
        for j in range(cols):
            cc=cells[(i,j)]
            if j==0:
                cc.set_facecolor(COR_ROXO); cc.set_text_props(color='white',weight='bold',ha='left')
            else:
                cc.set_facecolor('white' if i%2 else COR_CLARO)
            cc.set_edgecolor('white')
    plt.tight_layout(pad=0.5); plt.savefig(nome,dpi=300,bbox_inches='tight'); plt.close()
    print(f'✅ {nome}')

# ---------- 4. Leitura e pré-processamento -------------------
df = pd.read_csv('bomba_centro_cirurgico.csv')
df.rename(columns={'potencia':'potencia_ativa_kW','corrente':'corrente_A','data':'data_hora'}, inplace=True)
df['data_hora']=pd.to_datetime(df['data_hora'])
df.dropna(subset=['potencia_ativa_kW','corrente_A','tensao_saida','velocidade'], inplace=True)

df['potencia_aparente_kVA'] = (np.sqrt(3)*df['tensao_saida']*df['corrente_A'])/1000
df['fator_potencia'] = (df['potencia_ativa_kW']/df['potencia_aparente_kVA']).clip(0,1).fillna(0)
df['potencia_reativa_kVAr']=np.sqrt(np.maximum(0,df['potencia_aparente_kVA']**2-df['potencia_ativa_kW']**2))

oper = df[(df['potencia_ativa_kW']>1)&(df['tensao_saida']>100)].copy()
P_IN_NOM = P_SAIDA_KW/EF_NOM_MOTOR
oper['carreg_percent'] = oper['potencia_ativa_kW']/P_IN_NOM*100
oper['eficiencia']      = oper['carreg_percent'].apply(ef_interp)
oper['perdas_kW']       = oper['potencia_ativa_kW']*(1-oper['eficiencia'])
oper['pot_mec_kW']      = oper['potencia_ativa_kW']-oper['perdas_kW']
oper['torque_Nm']       = (oper['pot_mec_kW']*1000)/(oper['velocidade']*2*np.pi/60)

# ---------- 5. Tabelas --------------------------------------
tabela_png(pd.DataFrame({'Valor':['15 kW (20 cv)','~230 V (saída VFD)','92,8 %']},
                        index=['Potência Nominal','Tensão','Eficiência']),
           'Dados Nominais do Motor','tabela_nominais.png')

unit_map={'potencia_ativa_kW':'Potência Ativa [kW]',
          'corrente_A':'Corrente [A]',
          'fator_potencia':'Fator de Potência [-]',
          'potencia_aparente_kVA':'Potência Aparente [kVA]',
          'potencia_reativa_kVAr':'Potência Reativa [kVAr]',
          'carreg_percent':'Carregamento [%]',
          'torque_Nm':'Torque [Nm]'}
stats=(oper[['potencia_ativa_kW','corrente_A','fator_potencia',
             'potencia_aparente_kVA','potencia_reativa_kVAr',
             'carreg_percent','torque_Nm']]
       .describe().T.round(2)
       .rename(index=unit_map))
stats=stats[['mean','50%','std','min','max','25%','75%']]
stats.columns=['Média','Mediana','DP','Mín','Máx','Q1','Q3']
tabela_png(stats,'Estatísticas de Operação','tabela_stats.png')

# ---------- 6. Agrupamento 10 % ------------------------------
oper['bin']=pd.cut(oper['carreg_percent'],bins=np.arange(0,101,10),right=False)
binned=oper.groupby('bin',observed=False).agg(fp=('fator_potencia','mean'),
                                              ef=('eficiencia','mean'),
                                              perdas=('perdas_kW','mean'))
binned['center']=binned.index.map(lambda x: x.mid).astype(float)

# ---------- 7. Histograma de Carregamento -------------------
fig,ax=plt.subplots(figsize=(10,6))
ax.hist(oper['carreg_percent'],20,range=(0,100),color=COR_ROXO,edgecolor='white',alpha=0.9)
ax.axvline(oper['carreg_percent'].mean(),color=COR_TEAL,ls='--',lw=2.5,
           label=f"Média: {oper['carreg_percent'].mean():.1f}%")
ax.set_title('Histograma de Carregamento do Motor', fontsize=16, fontweight='bold', color=COR_CINZA)
ax.set_xlabel('Carregamento (%)'); ax.set_ylabel('Frequência'); safe_legend(ax,fontsize=11)
plt.tight_layout(); plt.savefig('histograma_carregamento.png',dpi=300); plt.close(); print('✅ histograma_carregamento.png')

# ---------- 8. Série I vs I_nom ------------------------------
serie_I=df.set_index('data_hora')['corrente_A'].resample('h').mean()
fig,ax=plt.subplots(figsize=(12,6))
ax.set_title('Série temporal – Corrente consumida vs. Corrente nominal',
             fontsize=16, fontweight='bold', color=COR_CINZA)
ax.plot(serie_I.index, serie_I, color=COR_ROXO, label='Corrente Real (A)')
ax.axhline(I_NOM_A, color=COR_TEAL, ls='--', lw=2, label=f'Corrente Nominal ({I_NOM_A} A)')
ax.set_ylabel('Corrente (A)'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m')); fig.autofmt_xdate()
safe_legend(ax,fontsize=11); plt.tight_layout(); plt.savefig('serie_temporal_corrente_nominal.png',dpi=300); plt.close(); print('✅ serie_temporal_corrente_nominal.png')

# ---------- 9. Eficiência vs Carga ---------------------------
fig,ax=plt.subplots(figsize=(10,6))
ax.plot(binned['center'], binned['ef']*100, color=COR_ROXO, marker='o', ls='--', label='Eficiência Média')
ax.scatter(CARG_PTS, EF_PTS*100, color=COR_TEAL, s=90, label='Catálogo WEG')
ax.add_patch(Rectangle((75,0),25,100, facecolor='green', alpha=0.15, label='Zona Ótima 75–100 %'))
ax.set_title('Curva de Eficiência vs. Carregamento', fontsize=16, fontweight='bold', color=COR_CINZA)
ax.set_xlabel('Carregamento (%)'); ax.set_ylabel('Eficiência (%)'); ax.set_ylim(85,100); safe_legend(ax,fontsize=11)
plt.tight_layout(); plt.savefig('eficiencia_zona_otima.png',dpi=300); plt.close(); print('✅ eficiencia_zona_otima.png')

# ---------- 10. FP vs Carga ----------------------------------
fig,ax=plt.subplots(figsize=(10,6))
ax.plot(binned['center'], binned['fp'], color=COR_TEAL, marker='o', ls='--', label='FP médio')
ax.set_title('Curva Fator de Potência vs. Carregamento', fontsize=16, fontweight='bold', color=COR_CINZA)
ax.set_xlabel('Carregamento (%)'); ax.set_ylabel('Fator de Potência'); safe_legend(ax,fontsize=11)
plt.tight_layout(); plt.savefig('fp_vs_carga.png',dpi=300); plt.close(); print('✅ fp_vs_carga.png')

# ---------- 11. Perdas vs Carga ------------------------------
fig,ax=plt.subplots(figsize=(10,6))
ax.plot(binned['center'], binned['perdas'], color=COR_ROXO, marker='o', label='Perdas médias (kW)')
ax.set_title('Curva de Perdas Totais vs. Carregamento', fontsize=16, fontweight='bold', color=COR_CINZA)
ax.set_xlabel('Carregamento (%)'); ax.set_ylabel('Perdas Totais (kW)'); safe_legend(ax,fontsize=11)
plt.tight_layout(); plt.savefig('perdas_vs_carga.png',dpi=300); plt.close(); print('✅ perdas_vs_carga.png')

# ---------- 12. Curva Torque-Velocidade ----------------------
s=np.linspace(0.001,1,200)
s_nom=(N_SINC-N_NOM)/N_SINC
s_max=s_nom*(T_MAX_TN+np.sqrt(T_MAX_TN**2-1))
T_curve=(2*T_MAX_TN*TORQUE_NOM)/(s/s_max+s_max/s)
n_curve=N_SINC*(1-s)
fig,ax=plt.subplots(figsize=(10,6))
ax.plot(n_curve, T_curve, color=COR_TEAL, ls='--', lw=2, label='Curva típica')
ax.scatter(oper['velocidade'], oper['torque_Nm'], color=COR_ROXO, alpha=0.25, s=15, label='Pontos reais')
ax.axhline(TORQUE_NOM, color='red', ls=':', label=f'Torque Nom. {TORQUE_NOM:.0f} Nm')
ax.set_title('Curva Característica Torque vs. Velocidade', fontsize=16, fontweight='bold', color=COR_CINZA)
ax.set_xlabel('Velocidade (RPM)'); ax.set_ylabel('Torque (Nm)')
ax.set_ylim(0, T_MAX_TN*TORQUE_NOM*1.1); safe_legend(ax,fontsize=11)
plt.tight_layout(); plt.savefig('curva_torque_velocidade.png',dpi=300); plt.close(); print('✅ curva_torque_velocidade.png')

# ---------- 13. Histogramas Potência & FP --------------------
fig,ax=plt.subplots(figsize=(10,6))
ax.hist(oper['potencia_ativa_kW'],30,color=COR_ROXO,edgecolor='white',alpha=0.9,label='Pot. ativa')
ax.axvline(oper['potencia_ativa_kW'].mean(),color=COR_TEAL,ls='--',lw=2.5,
           label=f"Média: {oper['potencia_ativa_kW'].mean():.2f} kW")
ax.set_title('Distribuição da Potência Ativa (Operação)', fontsize=16, fontweight='bold', color=COR_CINZA)
ax.set_xlabel('Potência (kW)'); ax.set_ylabel('Frequência'); safe_legend(ax,fontsize=11)
plt.tight_layout(); plt.savefig('histograma_potencia_color.png',dpi=300); plt.close(); print('✅ histograma_potencia_color.png')

fig,ax=plt.subplots(figsize=(10,6))
ax.hist(oper['fator_potencia'],30,color=COR_TEAL,edgecolor='white',alpha=0.9,label='FP')
ax.axvline(oper['fator_potencia'].mean(),color=COR_ROXO,ls='--',lw=2.5,
           label=f"Média: {oper['fator_potencia'].mean():.2f}")
ax.set_title('Distribuição do Fator de Potência (Operação)', fontsize=16, fontweight='bold', color=COR_CINZA)
ax.set_xlabel('Fator de Potência'); ax.set_ylabel('Frequência'); safe_legend(ax,fontsize=11)
plt.tight_layout(); plt.savefig('histograma_fp_color.png',dpi=300); plt.close(); print('✅ histograma_fp_color.png')

# ---------- 14. Séries temporais -----------------------------
daily_op=oper.set_index('data_hora')[['potencia_ativa_kW','fator_potencia']].resample('D').mean()
fig,ax1=plt.subplots(figsize=(12,6))
ax1.set_title('Desempenho Médio Diário (Apenas em Operação)', fontsize=16, fontweight='bold', color=COR_CINZA)
ax1.plot(daily_op.index, daily_op['potencia_ativa_kW'], color=COR_ROXO, marker='o', label='Potência (kW)')
ax1.set_ylabel('Potência (kW)', color=COR_ROXO); ax1.tick_params(axis='y', labelcolor=COR_ROXO)
ax2=ax1.twinx()
ax2.plot(daily_op.index, daily_op['fator_potencia'], color=COR_TEAL, marker='o', ls='--', label='FP médio')
ax2.set_ylabel('FP Médio', color=COR_TEAL); ax2.tick_params(axis='y', labelcolor=COR_TEAL); ax2.set_ylim(0.5,0.8)
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,l1+l2,fontsize=11,loc='best'); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m')); plt.gcf().autofmt_xdate()
plt.tight_layout(); plt.savefig('serie_temporal_em_operacao.png',dpi=300); plt.close(); print('✅ serie_temporal_em_operacao.png')

serie_full=df.set_index('data_hora')[['potencia_ativa_kW','fator_potencia']].resample('h').mean()
fig,ax1=plt.subplots(figsize=(12,6))
ax1.set_title('Série temporal – Potência e fp', fontsize=16, fontweight='bold', color=COR_CINZA)
ax1.fill_between(serie_full.index, serie_full['potencia_ativa_kW'], color=COR_TEAL, alpha=0.55, label='Pot. ativa')
ax1.plot(serie_full.index, serie_full['potencia_ativa_kW'], color=COR_TEAL, lw=1.2)
ax1.set_ylabel('Potência (kW)', color=COR_TEAL); ax1.set_ylim(0,12); ax1.tick_params(axis='y', labelcolor=COR_TEAL)
ax2=ax1.twinx()
ax2.plot(serie_full.index, serie_full['fator_potencia'], color=COR_ROXO, lw=2.4, label='fp')
ax2.set_ylabel('fp', color=COR_ROXO); ax2.tick_params(axis='y', labelcolor=COR_ROXO); ax2.set_ylim(0,1.05)
ax1.grid(which='major', axis='both', ls=':', color=COR_GRADE, lw=0.7)
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,l1+l2,fontsize=11,loc='best'); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m')); fig.autofmt_xdate()
plt.tight_layout(); plt.savefig('serie_temporal_ciclo_completo.png',dpi=300); plt.close(); print('✅ serie_temporal_ciclo_completo.png')

print('\n🎉  Processo finalizado – 12 PNGs gerados.')
