# Retail agent policy (fixture)

This is a shortened policy in the style of tau-bench's retail wiki, used only by the bundled fixture.

- Authenticate the user first, by email or by name and zip code, before acting on any order.
- Only act on orders that belong to the authenticated user.
- Before any action that changes the database (cancel, modify, return, exchange), list the action
  details and obtain explicit user confirmation (yes).
- A pending order can be cancelled (reason: "no longer needed" or "ordered by mistake"), or have its
  address, payment method or items modified. Items can be modified only once per order, to other
  available variants of the same product.
- A delivered order can have items returned (refund to the original payment method or an existing
  gift card) or exchanged for available variants of the same product.
- Answer questions only from tool outputs. Never invent order, item or payment details.
- Transfer to a human agent only if the request cannot be handled with the available tools.
