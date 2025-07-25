# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse
import asyncio
import importlib
import inspect
import pkgutil

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


@dataclass
class ParsedRemoteURL:
    """
    The parsed URL of the format:
        <host>:<port>[/path][?query]
    """

    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    path: Optional[str] = None
    query_params: Dict[str, List[str]] = field(default_factory=dict)


def parse_remote_url(url: str) -> ParsedRemoteURL:
    """
    Parses the remote URL into its constituent parts with support for:
    - Multiple hosts (comma-separated)
    - Path and query parameters in each host definition
    - Forward compatibility with legacy format

    Args:
        url: The URL to parse

    Returns:
        ParsedRemoteURL: The parsed URL components

    Raises:
        ValueError: If the URL is invalid
    """

    logger.debug(f"Parsing remote URL: {url}")
    parsed = urlparse(url)

    username = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port
    path = parsed.path if parsed.path else ""
    query = parse_qs(parsed.query) if parsed.query else {}

    return ParsedRemoteURL(
        host=host,
        port=port,
        path=path,
        username=username,
        password=password,
        query_params=query,
    )


class ConnectorContext:
    """
    Context for creating a connector.

    Attributes:
        url: The remote URL
        loop: The asyncio event loop
        local_cpu_backend: The local CPU backend
        config: Optional LMCache engine configuration
        parsed_url: Parsed representation of the URL
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig],
    ):
        self.url = url
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.config = config


class ConnectorAdapter(ABC):
    """Base class for connector adapters."""

    def __init__(self, schema: str):
        self.schema = schema

    @abstractmethod
    def can_parse(self, url: str) -> bool:
        """
        Check if this adapter can parse the given URL.
        """
        pass

    @abstractmethod
    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Create a connector using the given context.
        """
        pass


class ConnectorManager:
    """
    Manager for creating connectors based on URL.

    This class maintains a registry of connector adapters and creates
    the appropriate connector based on the URL.
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig] = None,
    ) -> None:
        self.context = ConnectorContext(
            url=url,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            config=config,
        )
        self.adapters: List[ConnectorAdapter] = []
        self._discover_adapters()

    def _discover_adapters(self) -> None:
        """Automatically discover and register all ConnectorAdapter subclasses."""
        # Import current package to ensure all modules are loaded
        # First Party
        import lmcache.v1.storage_backend.connector as connector_pkg

        # Discover all modules in the connector package
        for _, module_name, _ in pkgutil.iter_modules(connector_pkg.__path__):
            # Skip private modules and non-adapter modules
            if module_name.startswith("_") or not module_name.endswith("_adapter"):
                continue

            try:
                module = importlib.import_module(
                    f"{connector_pkg.__name__}.{module_name}"
                )

                # Find all ConnectorAdapter subclasses in the module
                for _, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, ConnectorAdapter)
                        and obj != ConnectorAdapter
                    ):
                        try:
                            adapter_instance = obj()
                            self.adapters.append(adapter_instance)
                            logger.info(f"Discovered adapter: {obj.__name__}")
                        except Exception as e:
                            logger.error(
                                "Failed to instantiate adapter "
                                f"{obj.__name__}: {str(e)}"
                            )
            except ImportError as e:
                logger.warning(f"Failed to import module {module_name}: {e}")

    def create_connector(self) -> RemoteConnector:
        for adapter in self.adapters:
            if adapter.can_parse(self.context.url):
                return adapter.create_connector(self.context)

        raise ValueError(f"No adapter found for URL: {self.context.url}")


def CreateConnector(
    url: str,
    loop: asyncio.AbstractEventLoop,
    local_cpu_backend: LocalCPUBackend,
    config: Optional[LMCacheEngineConfig] = None,
) -> Optional[InstrumentedRemoteConnector]:
    """
    Create a remote connector from the given URL.

    Supported URL formats:
    - redis://[[username]:[password]@]host[:port][/database][?option=value]
    - rediss://[[username]:[password]@]host[:port][/database][?option=value] (SSL)
    - redis-sentinel://[[username]:[password]@]host1:port1[,host2:port2,...]/service_name
    - lm://host:port
    - infinistore://host:port[?device=device_name]
    - mooncakestore://host:port[?device=device_name]
    - blackhole://[any_text]
    - audit://host:port[?verify=true|false]
    - fs://[host:port]/path

    Examples:
    - redis://localhost:6379
    - rediss://user:password@redis.example.com:6380/0
    - redis-sentinel://user:password@sentinel1:26379,sentinel2:26379/mymaster
    - lm://localhost:65432
    - infinistore://127.0.0.1:12345?device=mlx5_0
    - mooncakestore://127.0.0.1:50051
    - blackhole://
    - audit://localhost:8080?verify=true
    - fs:///tmp/lmcache
    - external://host:0/external_log_connector.lmc_external_log_connector/?connector_name=ExternalLogConnector

    Args:
        url: The remote URL
        loop: The asyncio event loop
        local_cpu_backend: The local CPU backend
        config: Optional LMCache engine configuration

    Returns:
        RemoteConnector: The created connector

    Raises:
        ValueError: If the connector cannot be created
    """

    # Basic URL validation - check for scheme
    if "://" not in url:
        raise ValueError(f"Invalid remote url {url}: missing scheme")

    manager = ConnectorManager(url, loop, local_cpu_backend, config)
    connector = manager.create_connector()

    return InstrumentedRemoteConnector(connector)
