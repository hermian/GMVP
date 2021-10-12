#!/usr/bin/env python
# coding: utf-8

# # GMVP2.0 (연금 포트 폴리오)
# https://blog.naver.com/myisiq999/222108832439
#
#  * csv파일의 헤더를 'Date,Close,RETURN'로 수정한다.



import bt
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from datetime import datetime, date, timedelta
import pprint

pd.options.display.float_format = '{:.2f}'.format


def get_data(start, end):
    """read_csv()에서 읽은 df 즉 기존 데이터의 마지막 날짜를 읽고 해당 날짜부터 end까지 데이타를 읽어서 append한 dataframe을 생성합니다."""
    # start = df.tail(1).index[0] + timedelta(days=1)

    ARIRANG신흥국MSCI합성H = fdr.DataReader("195980", start=start, end=end)['Close']
    HANAROKAP초장기국고채 = fdr.DataReader("346000", start=start, end=end)['Close'] # 2020-01
    KBSTAR국채선물10년 = fdr.DataReader("295000", start=start, end=end)['Close']
    KINDEX미국SP500 = fdr.DataReader("360200", start=start, end=end)['Close']
    KODEXWTI원유선물H = fdr.DataReader("261220", start=start, end=end)['Close']
    KODEX골드선물H = fdr.DataReader("132030", start=start, end=end)['Close']
    KODEX미국채울트라30년선물H = fdr.DataReader("304660", start=start, end=end)['Close']
    KODEX다우존스미국리츠H = fdr.DataReader("352560", start=start, end=end)['Close']
    TIGER라틴35 = fdr.DataReader("105010", start=start, end=end)['Close']
    TIGER미국채10년선물 = fdr.DataReader("305080", start=start, end=end)['Close']
    TIGER미국나스닥100 = fdr.DataReader("133690", start=start, end=end)['Close']
    TIGER일본니케이225 = fdr.DataReader("241180", start=start, end=end)['Close']
    TIGER미국다우존스30 = fdr.DataReader("245340", start=start, end=end)['Close']
    TIGER유로스탁스배당30 = fdr.DataReader("245350", start=start, end=end)['Close']


    df1 = bt.merge(ARIRANG신흥국MSCI합성H, HANAROKAP초장기국고채, KBSTAR국채선물10년, KINDEX미국SP500, KODEXWTI원유선물H, KODEX골드선물H, KODEX미국채울트라30년선물H,
                 KODEX다우존스미국리츠H, TIGER라틴35, TIGER미국채10년선물, TIGER미국나스닥100, TIGER일본니케이225, TIGER미국다우존스30, TIGER유로스탁스배당30)

    df1.columns = ['ARIRANG신흥국MSCI합성H', 'HANAROKAP초장기국고채', 'KBSTAR국채선물10년', 'KINDEX미국SP500', 'KODEXWTI원유선물H', 'KODEX골드선물H', 'KODEX미국채울트라30년선물H',
                 'KODEX다우존스미국리츠H', 'TIGER라틴35', 'TIGER미국채10년선물', 'TIGER미국나스닥100', 'TIGER일본니케이225', 'TIGER미국다우존스30', 'TIGER유로스탁스배당30']

    df1 = df1.dropna()
    df1.to_csv("gvmp_prices.csv")
    return df1

class WeighDaysAverageMomentumScoreSelectN(bt.Algo):
    '''
    평균모멘텀스코어를 구하고 상위 6개의 주식을 선택하여 평균모멘텀스코어에 대한 비중 할당.

    나머지는 현금을 보유?

    :Args:
        * n : 선택할 주식 갯수(평균모멘텀스코어 상위 n개 선택)
        * lookback : 일간평균모멘텀스코어를 구할 기간 (1년)
        * lag : 지연일 (1 or 0) 1이라면 전날 기준, 0이라면 당일 기준
                백테스팅할때는 1, 비중을 당일 중 끝나고 계산한다면 0을 설정
    '''
    def __init__(self, n=6):
        super(WeighDaysAverageMomentumScoreSelectN, self).__init__()
        # FIXME : 이상하게 12개월이 들어가면 처음에는 12개월로 할당 되지만 __call__에서 12 months, 1 year로 24개월로 개산됨
        self.n = n

    def calc_average_momentum_score(self, prc):
        average_momentum_score = pd.Series(dtype='float64')
        # prc = prc[-250:]
        # print(f"==={len(prc)}======prc\n{prc}")
        for c in prc.columns:
            # print(f"0 {prc[c][0]}, -1 {prc[c][-1]}")
            # average_momentum_score[c] = np.where(prc[c]>start_day, 1, 0).sum()/length
            average_momentum_score[c] = np.mean(np.where(prc[c][-1]>prc[c], 1, 0)[:-1])
            # return np.mean(np.where(x[-1]>x, 1, 0)[:-1]) # 당일 날짜 비교는 제외해준다 [:-1]

        return average_momentum_score

    def __call__(self, target):

        def _calc_average_momentum_score(c):
            return np.mean(np.where(c[-1]>c, 1, 0)[:-1])

        selected = target.temp['selected']
        # print(f"selected : {selected}")
        prc = target.universe.loc[:, selected].dropna()
        # print(prc)
        average_momentum_score = prc.apply(_calc_average_momentum_score)
        print(f"\n======= Average Momentum Score ======\n{average_momentum_score.sort_values(ascending=False)}")
        print("==========================================")

        average_momentum_score_select_n =average_momentum_score.nlargest(self.n)
        print(f"@@@@ 상위 {self.n}개\n{average_momentum_score_select_n}")
        #sort_values(ascending=False)[:self.n]

        weights = average_momentum_score_select_n.div(self.n)
        # print(f"\nweigthts : {list(weights.index)}, {weights.sum()} =>\n{weights}\n")
        target.temp['weights'] = pd.Series(weights, index=list(weights.index)) # 엔진에 비중할당
        target.temp['selected'] = list(weights.index) # 엔진에 선택 주식 할당
        return True


class WeighMinVolatility(bt.Algo):
    """ PyPortfolioOpt 패키지를 사용하여 min_volatility Optimizer로 비중을 계산"""
    def __init__(self):
        super(WeighMinVolatility, self).__init__()
        # FIXME : 이상하게 12개월이 들어가면 처음에는 12개월로 할당 되지만 __call__에서 12 months, 1 year로 24개월로 개산됨

    def __call__(self, target):
        selected = target.temp['selected']

        prc = target.universe.loc[:, selected]
#         print(prc)

        mu = expected_returns.mean_historical_return(prc)
        S = risk_models.sample_cov(prc, frequency=252)
        # S = risk_models.risk_matrix(prc, "ledoit_wolf", frequency=252)

        # Optimise for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.1))#!!! 원래 전체로 할때는 0.1로 제한
        # 하드 코딩이라 수정해야 한다.
#         bonds = ["HANAROKAP초장기국고채", "KBSTAR국채선물10년", "KODEX미국채울트라30년선물H", "TIGER미국채10년선물"]

#         for b in bonds:
#             if b in prc.columns:
#                 i = prc.columns.get_loc(b)
# #                 print(b, i)
#                 ef.add_constraint(lambda w: w[i] <= 0.05)

        #ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        raw_weights = ef.min_volatility()
        # print(f"raw_weights : {raw_weights}")
        cleaned_weights = ef.clean_weights()

        print("\n======= Min Vol=======")
        pprint.pprint(cleaned_weights)
        print("=================================")

        target.temp['weights'] = pd.Series(cleaned_weights)
        return True


if __name__ == "__main__":
    '''https://wikidocs.net/73785 argparse
    '''
    import argparse

    parser = argparse.ArgumentParser(description="GMVP2.0의 당일 종가 기준 비중을 계산하여 출력한다.\n"
                                                 "당일은 오후 3시 30분 이후에 수행해야 한다.",
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--date', '-d', type=str, dest = "date", action="store",
                    help="비중을 계산 하고자 하는 날짜를 적는다. 생략하면 당일 날짜를 할당(ex. -d 2021-09-30).\n"
                         "9월 30일 비중계산 다음날 이 비중대로 투자한다.")
    parser.add_argument('--lookback', '-l', type=int, dest = "lookback", action="store",
                    help="AMS/Min Volatility를 계산할 데이터 갯수(ex, -l 250).\n"
                         "해당옵션을 사용하지 않으면 1년 데이터 받아오고 적으면 해당 갯수만큼 사용한다.")
    parser.add_argument('--select', '-s', type=int, dest = "select", action="store", default=6,
                    help="AMS에서 선택할 모멘텀 상위 주식 갯수(ex, -s 5).\n"
                         "해당옵션을 사용하지 않으면 모멘텀 상위 6개를 선택")

    args = parser.parse_args()

    if args.date:
        now = args.date
        now = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        now = datetime.now()

    lookback = pd.DateOffset(years=1)
    if args.lookback:
        before_a_year = now - timedelta(args.lookback+200) # 넉넉하게 데이터를 받아온다.
    else:
        before_a_year = now.replace(year=now.year - 1) # 1년전
    # 비중을 구하기 위해 넣은 단순 초기값입니다. 실제 현재 나의 금액이 아닙니다
    INIT_CAP = 10000000000.0

    print(f"=====start : {before_a_year} \n===== end : {now}\n")
    data = get_data(start=before_a_year, end=now)
    print(data)
    if args.lookback:
        data = data[-args.lookback:]
    # data = pd.read_csv("gvmp_prices.csv", index_col=0, parse_dates=True)
    print(f"data length : {len(data)}")
    print(f"number of equity in MomentumScore : {args.select}")
    # print(data)


    MomScore = bt.Strategy(
            "MomScore",
            [
                # bt.algos.RunMonthly(run_on_end_of_period=True, run_on_last_date=True),
                bt.algos.RunOnDate(now.strftime("%Y-%m-%d")),
                bt.algos.SelectAll(),
                # bt.algos.SelectMomentum(n=5, lookback=pd.DateOffset(years=1)),
                WeighDaysAverageMomentumScoreSelectN(n=args.select),
                bt.algos.Rebalance()
            ]
        )

    MinVol = bt.Strategy(
            "MinVol",
            algos = [
                # bt.algos.RunMonthly(run_on_end_of_period=True,run_on_last_date=True),
                bt.algos.RunOnDate(now.strftime("%Y-%m-%d")),
                bt.algos.SelectAll(),
                WeighMinVolatility(),
                bt.algos.Rebalance()
            ]
        )

    st_pooled = bt.Strategy(
            "Pooled",
            algos = [
                # bt.algos.RunMonthly(run_on_end_of_period=True,run_on_last_date=True),
                bt.algos.RunOnDate(now.strftime("%Y-%m-%d")), #1년 뒤부터 시작 : 시작전 12개월 데이터 필요
                bt.algos.PrintDate(),
                bt.algos.SelectAll(),
                bt.algos.WeighSpecified(MomScore=0.5, MinVol=0.5), ### 비중을 조절해 줄수 있습니다.
                bt.algos.Rebalance()
            ],
            children= [MomScore, MinVol]
        )

    bt_pooled = bt.Backtest(st_pooled, data, initial_capital=INIT_CAP)
    r_pooled = bt.run(bt_pooled)

    weights = r_pooled.get_security_weights().loc[now.strftime("%Y-%m-%d")]
    # print(list(weights.columns))
    print(weights)
    # weights.to_csv("weights.csv")
    w = weights.T.sort_values(ascending=False)
    print("="*80)
    print(f"== 투자할 비중은 아래와 같습니다. ( {now.strftime('%Y-%m-%d')} )")
    print("="*80)
    print(w*100)
    cash = (1-w.sum())*100
    print("현금 : %.2f" % cash)
    # w.to_csv("w.csv")

