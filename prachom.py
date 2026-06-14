# Problem 2-1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import t, chi2



Teams=pd.read_csv("data/Teams.csv")


# 데이터 전처리: 팀-연도 데이터 생성
teams_2_1 = Teams[
    (Teams["yearID"] >= 2010) &
    (Teams["yearID"] <= 2025) &
    (Teams["yearID"] != 2020) &
    Teams["W"].notna() &
    Teams["L"].notna() &
    Teams["R"].notna() &
    Teams["RA"].notna() &
    (Teams["W"] + Teams["L"] > 0) &
    (Teams["R"] > 0) &
    (Teams["RA"] > 0)
].copy()

teams_2_1["G"] = teams_2_1["W"] + teams_2_1["L"]
teams_2_1["RS"] = teams_2_1["R"]
teams_2_1["WPct"] = teams_2_1["W"] / teams_2_1["G"]
teams_2_1["log_ratio"] = np.log(teams_2_1["RS"] / teams_2_1["RA"])
teams_2_1["logRS"] = np.log(teams_2_1["RS"])
teams_2_1["logRA"] = np.log(teams_2_1["RA"])
teams_2_1["logit_WPct"] = np.log(
    teams_2_1["WPct"] / (1 - teams_2_1["WPct"])
)

teams_2_1 = teams_2_1[
    [
        "yearID", "franchID", "teamID", "W", "L", "G",
        "RS", "RA", "WPct", "log_ratio",
        "logRS", "logRA", "logit_WPct"
    ]
]

teams_2_1 = teams_2_1[
    (teams_2_1["WPct"] > 0) &
    (teams_2_1["WPct"] < 1)
].copy()


# 1. Bill James 공식
def bill_james_formula(x, k):
    RS, RA = x
    return 1 / (1 + (RA / RS) ** k)


popt, pcov = curve_fit(
    bill_james_formula,
    (teams_2_1["RS"], teams_2_1["RA"]),
    teams_2_1["WPct"],
    p0=[2],
    bounds=(0, np.inf)
)

k_hat = popt[0]
print("k의 점추정량:", k_hat)

# 95% 신뢰구간
n = len(teams_2_1)
p = 1
df = n - p
se_k = np.sqrt(np.diag(pcov))[0]

k_ci = np.array([
    k_hat - t.ppf(0.975, df=df) * se_k,
    k_hat + t.ppf(0.975, df=df) * se_k
])

print("95% 신뢰구간:", k_ci)


# 2. 절편이 없는 로지스틱 회귀
X_ratio = teams_2_1[["log_ratio"]]
y = teams_2_1["WPct"]

glm_ratio = sm.GLM(
    y,
    X_ratio,
    family=sm.families.Binomial(),
    freq_weights=teams_2_1["G"]
).fit()

beta1_hat = glm_ratio.params["log_ratio"]
beta1_ci = glm_ratio.conf_int().loc["log_ratio"]

print("beta의 점추정량, 95% 신뢰구간:")
print(beta1_hat)
print(beta1_ci)
print("1의 결과와 매우 유사")


# 3. 모형적합결과 진단
teams_2_1["eta_ratio"] = glm_ratio.predict(X_ratio, linear=True)
teams_2_1["pred_ratio"] = glm_ratio.predict(X_ratio)
teams_2_1["pred_ratio_manual"] = 1 / (1 + np.exp(-teams_2_1["eta_ratio"]))

# Deviance residuals
teams_2_1["deviance_resid"] = glm_ratio.resid_deviance


# i. Residual deviance에 대한 해석
resid_dev = glm_ratio.deviance
resid_df = glm_ratio.df_resid
resid_dev_p = chi2.sf(resid_dev, df=resid_df)

print("3-i. Residual deviance diagnostic")
print("Residual deviance:", resid_dev)
print("Residual df:", resid_df)
print("Chi-square upper-tail p-value:", resid_dev_p)
print(
    "Small p-value means the model has lack of fit "
    "relative to the binomial GLM assumption.\n"
)


# ii. Deviance residuals vs linear predictors
plt.figure(figsize=(8, 5))
plt.axhline(0, color="gray", linewidth=0.5)
plt.scatter(
    teams_2_1["eta_ratio"],
    teams_2_1["deviance_resid"],
    alpha=0.75
)
plt.title("Deviance residuals vs linear predictors")
plt.xlabel(r"$\eta = \hat{\beta}_1 \log(RS / RA)$")
plt.ylabel("Deviance residual")
plt.tight_layout()
plt.show()

print(
    "그래프가 곡선 패턴이 뚜렷하거나 잔차가 한쪽으로 몰리지 않고 "
    "0 주변에 무작위로 흩어져 있는 모형이 된다."
)


# logRA와 logRS를 따로 넣은 모형
X_logs = teams_2_1[["logRA", "logRS"]]

glm_logs = sm.GLM(
    y,
    X_logs,
    family=sm.families.Binomial(),
    freq_weights=teams_2_1["G"]
).fit()

coef_logs = glm_logs.params
ci_logs = glm_logs.conf_int()

print("logRA, logRS 점추정량, 95% 신뢰구간:")
print(coef_logs)
print(ci_logs)


# 1, 2항의 추정 계수와 비교
estimate_compare_all = pd.DataFrame(
    {
        "estimate": [
            k_hat,
            beta1_hat,
            coef_logs["logRS"],
            -coef_logs["logRA"]
        ],
        "lower": [
            k_ci[0],
            beta1_ci[0],
            ci_logs.loc["logRS", 0],
            -ci_logs.loc["logRA", 1]
        ],
        "upper": [
            k_ci[1],
            beta1_ci[1],
            ci_logs.loc["logRS", 1],
            -ci_logs.loc["logRA", 0]
        ]
    },
    index=[
        "Bill James nls: k",
        "Logistic glm: beta1",
        "Logistic glm: logRS coef",
        "Logistic glm: -logRA coef"
    ]
)

print(estimate_compare_all)

print(
    "logRS coefficient + logRA coefficient:",
    coef_logs["logRS"] + coef_logs["logRA"]
)

print(
    "log(RS)와 log(RA)를 따로 넣어도, "
    "log(RS/RA) 하나를 넣은 모형과 거의 다르지 않다."
)






# ::: {.panel-tabset}

# ## R

# ```{r}

# ```

# ## Python

# ```{python}

# ```

# :::