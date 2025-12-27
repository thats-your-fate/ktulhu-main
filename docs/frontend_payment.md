## Frontend Payment Integration Guide

The backend now exposes a small Stripe Checkout helper. Follow these steps to integrate a “Go Premium” button in the frontend.

### 1. Configure the backend

Set the following environment variables before launching the server (a default `config/payment.env` is provided with Stripe test keys):

- `STRIPE_PUBLISHABLE_KEY` – the publishable key (`pk_...`) returned to the frontend via `/payment/config`.
- `STRIPE_SECRET_KEY` – your live/test secret key (`sk_...`).
- `STRIPE_PRICE_ID` – the exact price ID from Stripe for the subscription or one-off purchase.
- `STRIPE_SUCCESS_URL` – absolute URL where Stripe should redirect after payment (e.g. `https://app.example.com/payments/success`).
- `STRIPE_CANCEL_URL` – absolute URL for cancel flows (optional, defaults to `/payment/cancel`).

When these are present, the server prints `Stripe checkout enabled` on boot and serves `POST /payment/create-checkout-session`.

### 2. Call the checkout endpoint

From the frontend:

1. Fetch `/payment/config` to grab the publishable key (if you need to initialize Stripe.js).
2. Send a POST request to `/payment/create-checkout-session` when the user clicks “Upgrade”. No body is required:

```ts
async function startCheckout() {
  const res = await fetch('/payment/create-checkout-session', {
    method: 'POST',
    credentials: 'include', // keep session cookies/JWT if needed
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Checkout failed: ${text}`);
  }

  const { checkout_url } = await res.json();
  window.location.href = checkout_url;
}
```

The response JSON is `{ session_id, checkout_url }`. Redirect the browser to `checkout_url` immediately.

### 3. Handle the return routes

Stripe redirects to the URLs configured in the env vars. Build simple success/cancel pages in the frontend that:

1. Call `/external/api/profile` to refresh the user record.
2. If the backend reports the role as `paid` or `admin`, update the UI accordingly.

### 4. Activate the subscription

Stripe appends `session_id={CHECKOUT_SESSION_ID}` to the success URL (the backend injects this automatically if you forget). On your success page:

1. Read the `session_id` query parameter.
2. Send an authenticated POST to `/payment/activate` with `{ "session_id": "<id>" }`. Example:

```ts
async function activateSubscription(sessionId: string) {
  const res = await fetch('/payment/activate', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Activation failed: ${text}`);
  }

  return res.json(); // { user_id, role, updated }
}
```

3. After the call succeeds, re-fetch `/external/api/profile` to show the upgraded role. The backend only flips the user’s role to `paid` when `/payment/activate` confirms the Checkout session is marked as paid and that it belongs to the authenticated user.

If the call fails (expired session, mismatched user, etc.), show an error and let the customer retry from their account settings.

### 5. Optional: UI indicators

- Show the number of remaining generations using the `generation_limit` / `generations_remaining` properties already returned by `/external/api/profile` and `/external/api/generate`.
- Disable the Upgrade button when the backend reports payments are disabled (server responds `503 payments_not_configured`).

That’s it—your frontend only needs to trigger the checkout endpoint and handle the redirect. All sensitive Stripe logic stays on the backend.***
