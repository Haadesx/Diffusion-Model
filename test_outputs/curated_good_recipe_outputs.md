# Curated Good Recipe Outputs

Checkpoint: `best_step88000.pt`

These are the strongest samples from the local tests. They were selected for
recipe structure, ingredient relevance, and usable instructions.

## Chocolate Cake

Source report: `chocolate_cake_checkpoint_test_20260430_193246.md`

```text
Chocolate Cake

Ingredients:
- chocolate cake
- eggs
- oil
- butter
- sugar
- chocolate
- margarine
- powdered sugar
- water
- vanilla

Instructions:
1. Preheat oven to 350
2. Prepare cake mix as directed on box
3. Add the rest of ingredients, and mix well
4. Pour into greased and floured 9 x 13-inch pan
5. Bake 30 to 35 minutes
```

## Chocolate Cake Ingredients

Source report: `chocolate_cake_checkpoint_test_20260430_193248.md`

```text
- butter
- sugar
- eggs
- flour
- baking powder
- salt
- soda
- cocoa
- milk
- vanilla
- buttermilk
- coconut
```

## Macaroni and Cheese

Source report: `macaroni_and_cheese_checkpoint_test_20260430_193813.md`

```text
Macaroni and Cheese

Ingredients:
- elbow macaroni
- Cheddar cheese
- butter
- milk
- pimento
- sugar
- flour
- eggs
- salt
- pepper

Instructions:
1. Cook macaroni according to package directions
2. Drain
3. Mix in remaining ingredients
4. Pour into greased 1 1/2-quart casserole
5. Bake at 375 for 45 to 50 minutes.
```

## Meatloaf

Source report: `meatloaf_checkpoint_test_20260430_193841.md`

```text
Meatloaf

Ingredients:
- rice
- ground beef
- tomato paste
- tomato sauce
- eggs
- onion
- garlic
- basil
- salt
- pepper
- oregano

Instructions:
1. Cook the rice in salted water until just tender (about 15 minutes)
2. Drain
3. Mix the meat mixture with the remaining ingredients
4. Shape into a loaf
5. Bake in a 350 degree oven for 1 hour.
```

## Butter Chicken, Literal American Style

Source report: `butter_chicken_checkpoint_test_20260430_193633.md`

```text
Butter Chicken

Ingredients:
- Chicken
- Salt
- Butter
- Pepper

Instructions:
1. Flatten chicken breasts and season with salt and pepper
2. Dip chicken in melted butter, then dredge in flour, then in flour
3. Fry chicken breasts in hot oil until done
4. Serve with rice
```

## Spaghetti and Meatballs, Partial

Source report: `spaghetti_and_meatballs_checkpoint_test_20260430_193746.md`

This one is only partially good. It learned Italian-adjacent ingredients, but
not spaghetti or meatballs reliably.

```text
Spaghetti and Meatballs

Ingredients:
- butter
- fresh spinach
- marinara sauce
- Italian seasoned breadcrumbs
- parmesan cheese
- mozzarella cheese
- parsley

Instructions:
1. Preheat oven to 350 degrees
2. Melt butter in a 13 x 9-inch baking dish
3. Combine spinach, sauce, bread crumbs, cheese, Parmesan cheese and parsley
4. Pour over butter
5. Sprinkle with mozzarella
6. Bake at 375 degrees for 25 minutes.
```

## Weak Categories

These prompts drifted or produced generic dessert/casserole patterns:

- `chicken butter masala`
- `apple pie`
- `pancakes`
- `spaghetti and meatballs`, except the partial sample above

