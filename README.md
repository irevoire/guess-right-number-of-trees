This is a fork of the [vector-store-relancy-benchmark](https://github.com/meilisearch/vector-store-relevancy-benchmark) repo whose only purpose is to find the number of trees we must generate for a specified number of vectors + dimensions.


### The process

I ran the relevancy benchmark for multiple numbers of documents and trees. Here's the used command:
`cargo run --release -- --datasets db-pedia3-large --nb-trees 1,8,32,64,128,256,512,1024,2048,4096 --count 512,1024,2048,4096,8192,16_384,32_768,65_536,131_072,262_144,524_288,1_048_576`

The csv results are stored in https://github.com/irevoire/guess-right-number-of-trees/tree/main/results

From there, I made a few charts looking like that:
![image](https://github.com/user-attachments/assets/3446cf00-096b-400b-af8f-ff6e2c2524a0)

That chart is for 768 dimensions.
- Top to bottom is the number of vectors going from 512 to 4M vectors
- Left to right is the number of trees, going from 1 to 4096
- In the middle is the recall. Green is better.

From this table, I could extract the **minimum** number of trees required to get a certain recall.
I made a few charts to see what it would look like:
![image](https://github.com/user-attachments/assets/29c4799a-1a0b-42cd-beb3-79b90607c49b)

With this table I decided I wanted to maintain a recall between 0.8 and 0.9 with 0.9 when only have a few documents.
I handcrafted a new curve with the shape I wanted to follow:
![image](https://github.com/user-attachments/assets/b9f55805-2e8b-4625-99a7-de0d5cef1f64)

I repeated the process with 1536 and 3072 dimensions.

Finally ,@nnethercott found a formula following closely this curve:
![image](https://github.com/user-attachments/assets/7ad22e7f-68c2-45dc-8907-513445f04b4c)

That's the final version of the formula:
`= IF(A2 < 10000, 2 ^ (LOG($A2, 2) - 6), 2 ^ (LOG(A2, 10) + (768 / $C$3)^4))`

You can see the data we used for this analysis here: https://docs.google.com/spreadsheets/d/19Y1zgDokzLiw_q5iBfgII1E33XiryRyYUJJtfgRVoqE/edit?usp=sharing
These changes have been merged into arroy here: https://github.com/meilisearch/arroy/pull/138
