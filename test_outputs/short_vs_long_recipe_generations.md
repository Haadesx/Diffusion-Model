# Short vs Long Recipe Generations

This file compares the best current recipe-generation behavior from the newer checkpoints.

- Short/medium setting: `length=320`, `top_k=20`, `temperature=0.62`
- Long/detailed setting: `length=520`, `top_k=18`, `temperature=0.55`
- Best checkpoint for prompt-following so far: `best_step503000.pt`
- Best checkpoint for some cake completions: `checkpoint_step515000.pt`

The main result is that the model performs best as a structured recipe completer: title plus ingredients, then it generates instructions. Longer generation does not reliably mean more useful detail yet.

## Short / Medium Outputs

### Butter Chicken

Checkpoint: `best_step503000.pt`

Prompt style: title plus ingredients, then `<INSTR_START>`.

```text
Butter Chicken

Ingredients:
- chicken breast
- butter
- onion
- garlic
- ginger
- tomato sauce
- garam masala
- cumin
- chili powder
- heavy cream
- salt
- cilantro

Instructions:
1. Saute onion and garlic in butter until translucent
2. Add ginger and cook for 1 minute
3. Add tomato sauce and garam masala and stir well
4. Add cumin and chili powder
5. Stir constantly and bring to a boil
6. Cover and simmer for 15 minutes
7. Add cream and cook until sauce is reduced
8. Season to taste with salt and pepper
```

### Chocolate Cake

Checkpoint: `checkpoint_step515000.pt`

Prompt style: title plus ingredients, then `<INSTR_START>`.

```text
Chocolate Cake

Ingredients:
- flour
- sugar
- cocoa powder
- baking soda
- salt
- eggs
- milk
- vegetable oil
- vanilla
- boiling water

Instructions:
1. In the bowl bowl you want the cake
2. In a large bowl mix the first five ingredients
3. In a medium bowl mix the eggs, milk, oil and vanilla
4. Add the wet ingredients to the dry ingredients and stir till just combined
5. Pour the batter into a greased and floured 9 x 13 pan
6. Bake at 350 degrees for 30 to 35 minutes or until a toothpick comes out clean and comes out clean
7. Remove from oven and cool on racks
```

### Baked Macaroni And Cheese

Checkpoint: `best_step503000.pt`

Prompt style: title plus ingredients, then `<INSTR_START>`.

```text
Baked Macaroni and Cheese

Ingredients:
- elbow macaroni
- butter
- flour
- milk
- sharp cheddar cheese
- salt
- black pepper
- bread crumbs

Instructions:
1. In saucepan in a minimum of, salted water,, cook noodles until tender
2. Drain and set aside
3. In same saucepan, melt butter or margarine, stir in flour
4. Gradually add milk, stirring constantly
5. Bring to a boil
6. Stir in cheese, salt and pepper
7. Add cooked and cooked noodles
8. Cook until hot and cheese is melted
9. Pour into greased casserole dish
10. Top with bread crumbs
```

## Long / Detailed Outputs

### Detailed Butter Chicken

Checkpoint: `best_step503000.pt`

Prompt style: title plus longer ingredient list, then `<INSTR_START>`.

```text
Detailed Butter Chicken

Ingredients:
- boneless chicken breast
- butter
- onion
- garlic
- ginger
- tomato sauce
- garam masala
- cumin
- chili powder
- heavy cream
- salt
- black pepper
- cilantro
- cooked rice

Instructions:
1. Soak the chicken with water
2. Sprinkle with salt, pepper and ginger., then
3. > in
4. the garlic and butter in hot skillet
5. Add, curry, chili powder,, cumin, cream, and salt and pepper
6. _INSTR> Bring to a boil,, then simmer about 20 to 25 minutes
7. Serve Serve Serve over rice
```

### Detailed Chocolate Cake

Checkpoint: `best_step503000.pt`

Prompt style: title plus longer ingredient list, then `<INSTR_START>`.

```text
Detailed Chocolate Cake

Ingredients:
- all-purpose flour
- sugar
- cocoa powder
- baking soda
- baking powder
- salt
- eggs
- milk
- vegetable oil
- vanilla extract
- boiling water
- chocolate frosting

Instructions:
1. Combine all ingredients
2. Stir well
3. Pour into greased well-greased tube pan
4. Bake at 350 for 45 minutes
5. Top with a dollop of chocolate frosting
```

## Practical Takeaway

The short/medium generations are stronger than the long generations. The model can follow a structured ingredient-conditioned prompt, but it still struggles to sustain a long, detailed recipe without repetition, malformed control-token fragments, or recipe drift.

Best current sampling recipe:

```bash
python3 scripts/test_recipe_models.py \
  --profile recipe_poc_2day \
  --data_dir ./data_recipe_poc_2day \
  --checkpoint ./Model_Checkpoints/best_step503000.pt \
  --recipe "butter chicken" \
  --prefix "<RECIPE_START> <TITLE_START> Butter Chicken <TITLE_END> <INPUT_START> chicken breast <NEXT_INPUT> butter <NEXT_INPUT> onion <NEXT_INPUT> garlic <NEXT_INPUT> ginger <NEXT_INPUT> tomato sauce <NEXT_INPUT> garam masala <NEXT_INPUT> cumin <NEXT_INPUT> chili powder <NEXT_INPUT> heavy cream <NEXT_INPUT> salt <NEXT_INPUT> cilantro <INPUT_END> <INSTR_START>" \
  --num_samples 3 \
  --length 320 \
  --top_k 20 \
  --temperature 0.62
```
