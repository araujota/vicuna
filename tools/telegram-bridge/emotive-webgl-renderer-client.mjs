export async function getWebglRendererHealth(serviceUrl) {
  const url = new URL('/health', serviceUrl);
  const response = await fetch(url, {
    method: 'GET',
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`webgl renderer health failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return body;
}

export async function renderEmotiveAnimationViaWebglService(bundle, {
  serviceUrl,
  outputPath = '',
} = {}) {
  const url = new URL('/render', serviceUrl);
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      bundle,
      outputPath,
    }),
  });
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body?.error || `webgl renderer failed: ${response.status}`);
    error.stage = 'render';
    throw error;
  }
  return body;
}
