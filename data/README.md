# Data details
The data used in the paper contains two parts: **target order data** and **user history data**.

Target order data is a pickle file that stores a nested List. The first dimension of is the month, the second dimension is the day, the third dimension is the order item. 
For each item, the attributes are described as follows:

|Index|  Attribute        | Type       | remark   |
|---- |  ----             | ----       |----      |
|0    | UserID            | String     | -        |
|1    | Delivery Time     | Integer    | represented by 'xy', where x is the number of weekday and y is the number of time slot       |
|2    | Grid Coordinate x | Integer    | x Coordinate of the target destination out of 10    |
|3    | Grid Coordinate y | Integer    | y Coordinate of the target destination out of 10    |
|4    | Package Weight    | Integer    | -        |
|5    | Package Volume    | Integer    | -        |
|8    | Delivery Date    | DateTime    | -        |
|9    | Delivery Time Delta   | Integer    | The number of time slots elapsed after courier's arrival until customer's receiving        |
|10   | Is Autonomous Delivery   | Bool    | -        |
|11   | Is Order Re-delivered   | Bool    | -        |

The attributes in the user data are the same as above.