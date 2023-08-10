c.ServerApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors https://*.bodo.ai 'self' "
    }
}

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = False
c.ServerApp.port = 8080
c.ServerApp.trust_xheaders = True
c.ServerApp.allow_origin = '*'

# Allow removing default kernel
c.KernelSpecManager.ensure_native_kernel = False
