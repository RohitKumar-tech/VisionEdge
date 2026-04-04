"""
Sync service — builds face sync payloads for Edge Agents.
Edge Agents poll this every 60s to get added/updated/deleted persons.
"""
# TODO Phase 2: Implement


class SyncService:
    def get_pending_sync(self, site_id: str, client_id: str) -> list:
        """
        Return list of sync actions since last sync for this site.
        Actions: {action: 'add'|'update'|'delete', person_id, embedding_decrypted, ...}
        Embeddings are decrypted here and sent over TLS to Edge Agent.
        """
        raise NotImplementedError
