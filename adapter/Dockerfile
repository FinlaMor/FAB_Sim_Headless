# Adapter container.
#
# - PHP 8.1 (matches Talishar's pinned version per their CLAUDE.md).
# - shmop / pdo_mysql / redis / mbstring extensions so Talishar can boot
#   when ADAPTER_MODE=real.
# - Runs the built-in HTTP server on port 8000 by default.
#
# In production swap to php-fpm + nginx or to RoadRunner for higher RPS;
# for self-play workloads the built-in server with 4 workers easily
# saturates one CPU core, which is exactly what we want per game worker.
FROM php:8.1-cli-alpine

RUN apk add --no-cache --virtual .build-deps \
        autoconf gcc g++ make linux-headers \
        libpng-dev libjpeg-turbo-dev freetype-dev \
        oniguruma-dev libzip-dev \
    && docker-php-ext-configure gd --with-freetype --with-jpeg \
    && docker-php-ext-install -j$(nproc) \
        shmop pdo_mysql mysqli mbstring zip gd \
    && pecl install redis \
    && docker-php-ext-enable redis \
    && apk del .build-deps \
    && apk add --no-cache libpng libjpeg-turbo freetype libzip oniguruma

# OPcache makes a noticeable dent in cold-path cost; enable it.
RUN docker-php-ext-install opcache && \
    { \
      echo 'opcache.enable=1'; \
      echo 'opcache.enable_cli=1'; \
      echo 'opcache.memory_consumption=192'; \
      echo 'opcache.max_accelerated_files=8000'; \
      echo 'opcache.validate_timestamps=0'; \
      echo 'opcache.revalidate_freq=0'; \
      echo 'opcache.jit=tracing'; \
      echo 'opcache.jit_buffer_size=64M'; \
    } > /usr/local/etc/php/conf.d/opcache.ini

WORKDIR /srv
COPY . /srv/adapter

# Bring the Talishar clone in at build time. In docker-compose we override
# this with a bind-mount so the host clone stays the source of truth.
ENV ADAPTER_MODE=stub
ENV ADAPTER_DEBUG=0

EXPOSE 8000
CMD ["php", "-S", "0.0.0.0:8000", "-t", "/srv/adapter", "/srv/adapter/api.php"]
