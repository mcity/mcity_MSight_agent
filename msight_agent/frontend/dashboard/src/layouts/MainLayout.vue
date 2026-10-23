<template>
  <q-layout view="lHh Lpr lFf">
    <q-header elevated class="maize-underline">
      <q-toolbar>
        <q-icon name="hub" size="28px" class="q-mr-sm text-maize" />
        <q-toolbar-title>MSight Control Plane</q-toolbar-title>
      </q-toolbar>
    </q-header>

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script setup>
</script>

<style scoped>
.text-maize {
  color: #ffcb05;
}
.maize-underline {
  border-bottom: 3px solid #ffcb05;
}
</style>
