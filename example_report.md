# BUSINESS INTELLIGENCE

## COURSEWORK 1: Tior Games Case Study (Individual coursework)

## MODULE CODE & TITLE: M33150- FHEQ 7

## MODULE COORDINATOR: Dr Elisavet Andrikopoulou

## ASSESSMENT ITEM NUMBER: Item 1

## ASSESSMENT Title: Individual coursework

## DATE OF SUBMISSION: 26th March 2025

## Student No: UP


INTRODUCTION

This coursework focuses on analyzing and optimizing the player and spectator experience for Tior Games,
a gaming company renowned for its flagship title, League of Fun (LoF). As a Massive Online Battle Arena
(MOBA) game, LoF has a massive player base and hosts World Championships twice a year, drawing
professional players and global audiences.

Tior Games migrated its data infrastructure to a data warehouse on Amazon AWS, utilizing a constellation
schema that integrates multiple fact and dimension tables. As a junior developer, I have access to a
segment of this data warehouse, specifically focusing on past World Championships since 2016.

The objective of this coursework is to analyze the data, generate insights, and propose strategies that will:

Enhance the player and spectator experience
Boost the popularity of the World Championships
Improve profitability for Tior Games
To achieve this, I conducted the following tasks:

Queried the data warehouse using complex OLAP functions to extract meaningful insights.
Proposed enhancements to the schema by adding new dimensions and fact tables for deeper analysis.
Designed an ETL process to clean and integrate new data sources before loading them into the data
warehouse.

This coursework provides a structured approach to leveraging data-driven decision-making, ensuring that
Tior Games can optimize its competitive events and strengthen its market presence.


## Table of Contents

- BUSINESS INTELLIGENCE
- INTRODUCTION
- TASK
- ǪUERY 01: TOP 3 PLAYERS BY TOTAL KILLS FOR EACH YEAR
- ǪUERY 02: MERCHANDISE STOCK VS SALES ANALYSIS
- ǪUERY 03: ANALYZING PROMOTION EFFECTIVENESS: TICKET SALES, COSTS, AND REVENUE INSIGHTS
- ǪUERY 04: STADIUM EVENT ANALYSIS: IDENTIFYING THE MOST ACTIVE VENUES
- ǪUERY 06: PIVOTING EVENT GAMES BY YEAR ǪUERY 05: ANALYZING REFUNDS BY TYPE WITH RANKING AND CURRENCY FORMATTING G
- TASK
- TASK 2 PROPOSE ENHANCEMENTS TO THE SCHEMA
- GAMEPERFORMANCEFACT
- MAPDIM
- PLAYERPOPULARITYDIM
- PLAYERHEALTHDIM
- PLAYERGAMESPLAYEDDIM
- ETL AND DATA INTEGRATION
- ETL DATA SOURCE
- DATA CLEANING PROCESS FOR GAMEFACT TABLE: A STEP-BY-STEP GUIDE
- INTRODUCTION
- DATA CLEANING STEPS
- FINAL RESULT
- CONCLUSION
- ETL DATA SOURCE
- INTRODUCTION
- DATA TRANSFORMATION AND CLEANING STEPS
- FINAL RESULT
- CONCLUSION
- THE SǪL ǪUERY TO INSERT DATA
- REFERENCES


# TASK 01

## Ǫuery 01: Top 3 Players by Total Kills for Each Year

**Unique View:**

This query focuses on tracking the top players by kills each year. It's designed to identify and rank players
who dominate in terms of kills, showcasing their yearly achievements. The query extracts and ranks
players based on their performance, allowing you to easily spot who stood out each year.

**Query:**

WITH PlayerYearStats AS (
SELECT d.DateYear, p.PlayerGameName,
SUM(pr.PRKills) AS TotalKills,
RANK() OVER (PARTITION BY d.DateYear ORDER BY SUM(pr.PRKills) DESC) AS Rank
FROM PlayerDim p
JOIN PlayerInGameDim pig ON p.PlayerID = pig.PlayerID
JOIN PersonalRecordDim pr ON pig.PRID = pr.PRID
JOIN DateDim d ON pig.GameID = d.DateID
GROUP BY d.DateYear, p.PlayerGameName
)
SELECT DateYear, PlayerGameName, TotalKills, Rank
FROM PlayerYearStats
WHERE Rank <= 3
ORDER BY DateYear DESC, Rank ASC;

**Output Screenshot:**


**Explanation**

This SQL query works by first collecting data for each player’s kills, grouped by year. It uses a **Common
Table Expression (CTE)** to calculate the total kills for each player and ranks them within each year using
the RANK() function. The query then filters to select only the top 3 players for each year (WHERE Rank
<= 3), ensuring that you get a snapshot of the leading players. Finally, the results are ordered by year in
descending order, with the best-ranked players appearing first within each year.

The output provides a clean, year-by-year view of the top performers, offering insights into who
consistently dominated in terms of kills. This can be particularly useful for analysts, coaches, or teams
looking to track standout players over time.


Ǫuery 02: Merchandise Stock vs Sales Analysis

**Unique View**

This query provides an analysis of the stock, sales, and remaining stock for each merchandise type,
utilizing the CUBE function to create subtotals for different combinations of merchandise types.

**Query:**

SELECT
COALESCE(m.MerchandiseType, 'Total') AS MerchandiseType,
SUM(osf.MerchandiseStocked) AS TotalStock,
SUM(osf.MerchandiseSold) AS TotalSold,
(SUM(osf.MerchandiseStocked) - SUM(osf.MerchandiseSold)) AS StockLeft
FROM OnlineSalesFact osf
JOIN MerchandiseDim m ON osf.MerchandiseID = m.MerchandiseID
GROUP BY cube(m.MerchandiseType)
ORDER BY
m.MerchandiseType DESC;

**Output Screenshot:**

**Explanation:**

This query calculates the total stock, total sales, and remaining stock for each merchandise type, using
CUBE to create subtotals. It groups the data by merchandise type and orders the results in descending
order, showing the total sales performance across different categories and the remaining stock. The
COALESCE function replaces null values with "Total."


Ǫuery 03: Analyzing Promotion Effectiveness: Ticket Sales, Costs, and

Revenue Insights

**Unique View**

This query calculates the total tickets sold, promotion cost, and promotion revenue for each promotion
type, and ranks them based on total tickets sold, using the DENSE_RANK function.

WITH PromotionSummary AS (
SELECT
COALESCE(p.PromotionType,'Total') as PromotionType,
SUM(e.TicketsSold) AS TotalTicketsSold,
SUM(e.PromotionCost) AS TotalPromotionCost,
SUM(e.PromotionRevenue) AS TotalPromotionRevenue
FROM EventFact e
JOIN PromotionDim p ON e.PromotionID = p.PromotionID
GROUP BY CUBE(p.PromotionType)
)
SELECT
ps.PromotionType,
ps.TotalTicketsSold,
ps.TotalPromotionCost,
ps.TotalPromotionRevenue,
DENSE_RANK() OVER (ORDER BY ps.TotalTicketsSold DESC) - 1 AS Rank
FROM PromotionSummary ps
ORDER BY
CASE WHEN ps.PromotionType = 'Total' THEN 1 ELSE 0 END

**Output Screenshot:**


**Explanation**

This query first calculates the total tickets sold, total promotion cost, and total promotion revenue for each
promotion type using the CUBE function to create subtotals. Then, the DENSE_RANK function is used
to rank the promotion types based on total tickets sold, with the rank starting from 0. The COALESCE
function ensures that null values are replaced with 'Total'. The final result is ordered so that the "Total"
row appears last.

Ǫuery 04: Stadium Event Analysis: Identifying the Most Active Venues

**A Unique View**

This query summarizes the total events held in each stadium and ranks them based on the number of
events, with a separate row for the total events.

**Query:**

WITH StadiumSummary AS (
SELECT
COALESCE(s.StadiumName, 'Total') AS StadiumName,
COUNT(gf.EventID) AS TotalEvents
FROM GameFact gf
JOIN StadiumDim s ON gf.StadiumID = s.StadiumID
GROUP BY GROUPING SETS ( (s.StadiumName), () )
)

SELECT
ss.StadiumName,
ss.TotalEvents,
DENSE_RANK() OVER (ORDER BY ss.TotalEvents DESC) - 1 AS Rank
FROM StadiumSummary ss
ORDER BY
CASE WHEN ss.StadiumName = 'Total' THEN 1 ELSE 0 END,
ss.TotalEvents DESC;

**Output Screenshot:**


**Explanation:**

This query first calculates the total events held in each stadium using the GROUPING SETS function to
handle multiple aggregation levels, including a total row. Then, the DENSE_RANK function is applied
to rank the stadiums based on the total number of events, with the rank starting from 0. The COALESCE
function ensures that null values are replaced with 'Total' for the final row, and the results are ordered with
the "Total" row displayed last.

Ǫuery 05: Analyzing Refunds by Type with Ranking and Currency Formatting

**Unique View**

This query provides an overview of refunds by type, ranks them based on total refund amounts, and
formats the results in a user-friendly currency format.

**Query:**

WITH RefundSummary AS (
SELECT
COALESCE(r.RefundType, 'Total') AS RefundType,
SUM(rf.TicketsRefundedPND + rf.MerchandiseRefundedPND) AS TotalRefundAmount
FROM RefundFact rf
JOIN RefundDim r ON rf.RefundID = r.RefundID
GROUP BY ROLLUP (r.RefundType)
),
RankedRefundSummary AS (
SELECT
RefundType,
TotalRefundAmount,


ROW_NUMBER() OVER (ORDER BY TotalRefundAmount DESC) - 1 AS RefundRank
FROM RefundSummary
)
SELECT
RefundType,
FORMAT(TotalRefundAmount, 'C', 'en-GB') AS TotalRefundAmountFormatted,
RefundRank
FROM RankedRefundSummary
ORDER BY
CASE WHEN RefundType = 'Total' THEN 1 ELSE 0 END,
TotalRefundAmount DESC;

**Output Screenshot:**

**Explanation**

This query calculates and ranks refunds by type, while also formatting the refund amounts in a readable
currency format. Here's the breakdown:

- RefundSummary: It aggregates the total refund amount for each refund type (tickets and
    merchandise), using the ROLLUP function to include a total row.
- RankedRefundSummary: Applies ROW_NUMBER() to rank the refund types by the total refund
    amount, with the rank starting from 0.
- FORMAT: The FORMAT function is used to display the refund amounts in a currency format for
    easier reading, specifically using the British Pound format ('en-GB').


- ORDER BY: Orders the results, placing the 'Total' refund type at the bottom, followed by
    individual refund types sorted by total amount in descending order.

Ǫuery 06: Pivoting Event Games by Year

**A Unique View**

This query aggregates the count of games for each year, pivoting the data to show event counts across
multiple years and calculates a total across all years.

**Query:**

WITH EventGames AS (
SELECT
e.EventYear,
g.GameID
FROM GameFact g
JOIN EventDim e ON g.EventID = e.EventID
)
SELECT *,
([2016] + [2017] + [2018] + [2019] + [2020] + [2021] + [2022]) AS Total
FROM (
SELECT EventYear, GameID
FROM EventGames
) AS SourceTable
PIVOT (
COUNT(GameID)
FOR EventYear IN ([2016], [2017], [2018], [2019], [2020], [2021], [2022])
) AS PivotTable;

**Output Screenshot:**


**Explanation**

This query counts the number of games played each year for a specific event, using a pivot table. The
PIVOT operation transforms the data from a long format (with separate rows for each year) into a wide
format (with separate columns for each year). Additionally, the Total column sums the game counts for
each event across all years, giving a summary of total games for the entire period (2016–2022). This is
particularly useful for understanding the event's popularity and game count across multiple years.


# TASK 02

## Task 2 Propose Enhancements to The Schema

**Fact Table** :

## GamePerformanceFact

This table captures the measurable, quantitative metrics for game performance.

```
Fact Table: GamePerformanceFact
```
```
Column Name Data Type Description Key
```
GameID INT (^) Unique identifier for each game instance **PK**
PlayerID INT Foreign key linking to PlayerDim (player information) (^) **FK**
MapID INT Foreign key linking to MapDim (map-related details) (^) **FK**
HealthStatusID INT (^) Foreign key linking to PlayerHealthDim **FK**
GameDuration INT (^) Total duration of the game in minutes
Kills INT (^) Number of kills by the player during the game
Deaths INT (^) Number of times the player died in the game
Score DECIMAL(10, 2) (^) Player’s overall score in the game
PopularityID INT Foreign key linking to PlayerPopularityDim (^) **FK**
GamesPlayedID INT Foreign key linking to PlayerGamesPlayedDim (^) **FK
Target Audience:**

- **Game Analysts** : To measure and analyze game performance metrics.
- **Developers** : To improve game mechanics based on player behavior and map usage.
- **Marketing Teams** : To assess how popularity impacts performance.
- **Health Researchers** : To examine the relationship between player health and in-game
    performance.


**Possible Uses:**

- Analyze player performance by tracking kills, deaths, and game duration.
- Evaluate how different maps impact player outcomes (e.g., urban vs. jungle maps).
- Assess player popularity trends and their influence on game engagement.
- Examine how player health affects performance (e.g., higher scores for fit players).
- Identify top players based on their performance metrics (score, kills, deaths).

Dimension Table:

MapDim

Details about the map on which the game was played.

```
MapDim
```
```
Column Name Data Type Description Key
```
MapID INT (^) Unique identifier for each map **PK**
MapName VARCHAR(100) (^) Name of the map (e.g., Desert Arena)
MapType VARCHAR(50) (^) Type of map (e.g., Snow, Jungle, Urban)
MapSize VARCHAR(50) (^) Size of the map (e.g., Small, Medium)
Region VARCHAR(100) (^) Geographical region of the map
Target Audience:

- **Game Designers** : To understand which types of maps (urban, jungle, etc.) are most played.
- **Event Organizers** : To decide which maps are best suited for competitive tournaments.
- **Level Design Teams** : To optimize map layout and size based on player engagement.

Possible Uses:

- Track which map types and sizes are associated with better player performance.
- Analyze player preferences for certain map environments (e.g., snowy vs. desert maps).
- Study regional differences in map engagement (e.g., are certain maps more popular in different
    regions?).
- Improve balance by adjusting map layouts that may favor specific playstyles.


Dimension Table:

PlayerPopularityDim

```
PlayerPopularityDim
```
```
Column Name Data Type Description Key
```
PopularityID INT (^) Unique identifier for popularity records **PK**
PlayerID INT Foreign key linking to PlayerDim (^) **FK**
SocialMediaScore DECIMAL(5,2) (^) Popularity score based on social media followers
FanVotes INT (^) Number of votes from fans in popularity polls
PopularityRank INT (^) Overall popularity rank
Target Audience:

- **Marketing Teams** : To promote top players and leverage their popularity for brand endorsements.
- **Player Community Managers** : To engage with fan-favorite players and boost community
    interaction.
- **Tournament Organizers** : To prioritize popular players for competitive matches and public
    events.

Possible Uses:

- Track changes in player popularity based on social media activity, fan votes, and overall ranking.
- Identify rising stars and top-performing players for marketing campaigns.
- Analyze the impact of player popularity on game engagement and ticket sales.
- Encourage fans to vote for players, boosting player engagement and brand loyalty.


Dimension Table:

PlayerHealthDim

```
PlayerHealthDim
```
```
Column Name Data Type Description Key
```
HealthStatusID INT (^) Unique identifier for health records **PK**
PlayerID INT Foreign key linking to PlayerDim (^) **FK**
HealthCondition VARCHAR(50) (^) Health status description (e.g., Fit, Injured)
HeartRate DECIMAL(5,2) (^) Average heart rate during gameplay
InjuryType VARCHAR(100) (^) Type of injury (if any)
Target Audience:

- **Health Researchers** : To study the impact of player health on gaming performance.
- **Game Developers** : To design health-based challenges or create features that support healthy
    gameplay.
- **Fitness Trainers** : To advise eSports players on maintaining good health to improve performance.

Possible Uses:

- The PlayerHealthDim table can be used to analyze how player health, such as fitness levels,
    injuries, and heart rate, affects in-game performance. Researchers and health experts can track
    player health trends over time to understand how specific conditions (like injury or fatigue) impact
    gameplay. Developers can use these insights to create health-conscious game features, such as
    recovery periods for players or stress-reducing gameplay elements. Fitness trainers can use the
    data to advise eSports players on maintaining optimal health for peak performance. This data can
    also help design healthier and more balanced gaming environments.


Dimension Table:

PlayerGamesPlayedDim

```
PlayerGamesPlayedDim
```
```
Column Name Data Type Description Key
```
GamesPlayedID INT (^) Unique identifier for player-game records **PK**
PlayerID INT Foreign key linking to PlayerDim (^) **FK**
GameType VARCHAR(50) (^) Type of game played (e.g., Shooter, RPG)
TotalGamesPlayed INT (^) Total number of games played by the player
Target Audience:

- **Game Developers** : To track which types of games players are engaging with the most.
- **Player Engagement Teams** : To encourage players to try different game genres.
- **eSports Analysts** : To evaluate the versatility of professional players across different game types.

Possible Uses:

- Track the total number of games played by each player and their experience level.
- Identify which game types are most popular (e.g., shooters, RPGs, strategy games).
- Encourage players to diversify their gameplay and explore different game genres.
- Evaluate how experience in one game type (e.g., shooters) impacts performance in others (e.g.,
    RPGs).


# ETL AND DATA INTEGRATION

## ETL Data Source 1

## Data Cleaning Process for GameFact Table: A Step-by-Step Guide

## INTRODUCTION

To ensure the integrity of the data being loaded into the **GameFact** table within the Tior Games data
warehouse, I carried out a comprehensive data cleaning process. The aim was to eliminate any incomplete
or erroneous records, leaving only valid and reliable data for analysis. I used **Microsoft Excel** to clean the
dataset, which allowed for effective filtering, sorting, and validation.

## DATA CLEANING STEPS

**Step 1: Loading Data into Excel**

The first step involved importing the dataset into **Microsoft Excel**. This allowed me to leverage Excel’s
features, such as filtering and sorting, to better understand the structure of the data. Importing into Excel
also made it easier to perform validation and transformations to ensure the quality of the data before any
further analysis.

**Step 2: Applying Filters**

To begin cleaning, I applied filters to all the columns of the dataset. This helped in quickly identifying
and reviewing rows with missing or invalid values. The filtering function allowed me to sort and examine
the data systematically, which made the next steps of validation and modification more efficient.

**Step 3: Removing Null Values**

A major part of the data cleaning process involved identifying and eliminating rows with **null values**.
Since the **GameFact** table requires complete records to ensure proper analysis and reporting, any row
containing missing values in any column was removed. This significantly reduced the dataset, as many
records contained incomplete information that would not be suitable for loading into the data warehouse.


**Step 4: Validating Data Types and References**

Next, I checked the **data types** and **references** in each column to ensure that the dataset met the expected
format. This step included confirming that numeric fields did not contain text or special characters and
verifying that references matched existing entries in the relevant lookup tables. Any discrepancies were
addressed by removing or correcting the records.

## FINAL RESULT

After completing all the cleaning steps, the result was clear: **no valid records remained** in the dataset.
This outcome revealed that every row in the original dataset had at least one issue, whether it was missing
values, incorrect data types, invalid references, or outdated season data. As a result, it was determined that
no data could be loaded into the **GameFact** table.

**Output Screenshot:**

## CONCLUSION

The data cleaning process, although thorough, highlighted the quality of the initial dataset—unfortunately,
the data provided did not meet the necessary standards for integration into the warehouse. This serves as
a reminder of the importance of high-quality data sources in ensuring smooth data operations. Moving
forward, efforts to improve data collection and validation processes will be crucial to avoid such issues in
future datasets.


ETL Data Source 2

## INTRODUCTION

As part of the data preparation for loading into the **GameFact** table of the Tior Games data warehouse, I
undertook a meticulous data cleaning procedure. The goal was to filter out invalid, incomplete, or
irrelevant data and ensure only the most accurate and complete records were loaded into the system. The
process involved the use of both **Microsoft Excel** and a **JSON to Excel converter** tool (Minifier) to
ensure the data was properly structured and ready for further analysis. At the end of this process, only one
valid record was retained, indicating the original dataset contained significant errors or missing
information.

## DATA TRANSFORMATION AND CLEANING STEPS

**Step 1: Converting JSON to Excel**

The dataset was initially in **JSON** format, which is not ideal for spreadsheet-based analysis. To transform
it into a more manageable format, I used the **Minifier website** (https://www.minifier.org/json-to-excel) to
convert the JSON file into an **Excel file**. This conversion step was crucial in making the data tabular and
much easier to work with.

**Step 2: Importing Data into Excel**

Once the dataset was converted, I imported it into **Microsoft Excel** for further cleaning and validation.
Excel’s features such as sorting, filtering, and validation functionalities provided an efficient environment
for thoroughly reviewing and preparing the data.

**Step 3: Applying Filters**

To facilitate a more streamlined cleaning process, I applied **filters** to each column. This allowed me to
quickly identify and review specific rows based on criteria such as missing values, incorrect formats, or
any obvious inconsistencies in the data.

**Step 4: Removing Null Values**

One of the critical requirements for data insertion into the **GameFact** table was to ensure no field
contained **null values**. I reviewed each column to ensure completeness and removed all rows where any
field was blank or missing. This significantly reduced the dataset, as many records contained incomplete
information that could not be used.


**Step 5: Eliminating Invalid Data**

After handling null values, I further scrutinized the dataset to remove any **invalid data**. Specific checks
included:

- **Incorrect Data Types** : Records with mismatched data types (e.g., text in numeric fields) were
    discarded.
- **Invalid Dates** : I validated that all date fields were correct, ensuring proper formatting and
    accuracy, including checking for leap years and valid ranges.

## FINAL RESULT

After completing the data transformation and cleaning steps, only **one valid record** remained. This
indicates that the original dataset contained multiple issues, such as missing values, incorrect formats, and
outdated or invalid data. As a result, the majority of the dataset could not be loaded into the **GameFact**
table, as it did not meet the necessary standards for integration.

**Output Screenshot:**

**Conclusion**

The cleaning process highlighted the importance of starting with a high-quality dataset. Although the
original data had significant issues, the steps taken ensured that only valid, clean records were considered
for loading into the data warehouse. Moving forward, ensuring better data quality and validation from the
source will be crucial to avoid such extensive cleaning requirements in future datasets.


The SǪL Ǫuery To Insert Data

INSERT INTO GameFact

(

EventID, StadiumID, RefereeID, GameID, TimeID, DateID, PauseID, HighlightID,

```
GameDuration, GameNumberOfPause, GameTeam, GameInterruption, GameMinuteOfPause,
```
```
GameDurationOfPause, GameResult
```
)

VALUES

(

1, 10, 10, 5,89, 4674, 1500, 2, 10, 34, 'blue', 1, 24, 23, 'blue wins'

);


REFERENCES

University of Portsmouth Moodle- SQL Queries and Scripts, presentations slides and Relationships.
https://moodle.port.ac.uk/course/view.php?id=25671

For a starting point in learning SQL, I turned to the Khan Academy resource, specifically their 'Intro to
SQL' section
https://www.khanacademy.org/computing/hour-of-code/hour-of-code-lessons/hour-of-sql/v/welcome-to-
sql

To enhance my practical SQL skills, I utilized the interactive tutorials and examples provided by
w3schools.com
https://www.w3schools.com/sql/

To convert JSON data into Excel files, I used the JSON to Excel converter found at Minifier.org
https://www.minifier.org/json-to-excel


