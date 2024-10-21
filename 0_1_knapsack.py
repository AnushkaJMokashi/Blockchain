
import sys


def knapsack(wt, val, n, W):
  
    dp = [[0 for i in range(W + 1)] for j in range(n)]
    

    for i in range(wt[0], W + 1):
        dp[0][i] = val[0]
        
 
    for ind in range(1, n):
        for cap in range(W + 1):
       
            not_taken = 0 + dp[ind - 1][cap]
            

            taken = -sys.maxsize
            if wt[ind] <= cap:
                taken = val[ind] + dp[ind - 1][cap - wt[ind]]
                

            dp[ind][cap] = max(not_taken, taken)
    

    return dp[n - 1][W]

def main():
    wt = [1, 2, 4, 5]
    val = [5, 4, 8, 6]
    W = 5
    n = len(wt)
                                 
    print("The Maximum value of items the thief can steal is", knapsack(wt, val, n, W))

if __name__ == "__main__":
    main()

