# Data details
We provide the processed Beijing and Changshu datasets that are used in the paper.
The dataset contains two parts: **order data** and **history data**.


Target order data is a pickle file that stores a nested List. The first dimension of is the month, the second dimension is the day, the third dimension is the order item. 
For each item, the attributes are described as follows:

|Index|  Attribute        | Type       | remark   |
|---- |  ----             | ----       |----      |
|0    | UserID            | String     | Random ID number        |
|1    | Receiving Time     | Integer    | Represented by 'xy', where x is the number of weekday and y is the number of time slot       |
|2    | Grid Coordinate x | Integer    | x Coordinate of the target destination out of 10    |
|3    | Grid Coordinate y | Integer    | y Coordinate of the target destination out of 10    |
|4    | Weight    | Integer    | Normalized value        |
|5    | Volume    | Integer    | Normalized value        |
|8    | Date    | DateTime    | -        |
|9    | TimeDelta   | Integer    | The number of time slots elapsed between the courier's arrival and the customer's receiving        |
|10   | Is Autonomous   | Bool    | -        |
|11   | Is Re-delivered   | Bool    | -        |

The attributes in the history data are the same as above.
